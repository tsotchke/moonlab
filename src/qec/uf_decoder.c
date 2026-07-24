/**
 * @file uf_decoder.c
 * @brief Union-find decoder over a detector error model.
 *
 * Clusters grow outward from lit detectors until each has even parity or
 * has reached the boundary; the grown region is then peeled from its leaves
 * inward to produce a correction.  Both phases are linear in the size of the
 * grown region rather than in the size of the graph, which is why the cost
 * tracks the number of defects rather than the code size.
 */
#include "uf_decoder.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Growth units per multiple of the cheapest edge weight.  Higher keeps more
 * of the weight ordering; growth rounds scale with it, so it trades decode
 * time for accuracy. */
#ifndef MOONLAB_UF_WSCALE
#define MOONLAB_UF_WSCALE 8u
#endif

/* Shots per transpose tile; sized so one tile of syndromes stays in cache. */
#define UF_TILE 256u

/* Largest cluster (in defects) resolved by exact subset-DP matching; above
 * this the peel is used instead.  The DP table is 1<<UF_MATCH_CAP entries, so
 * this bounds per-thread scratch.  Clusters this large are vanishingly rare
 * at sub-threshold error rates. */
#define UF_MATCH_CAP 16u

/* Largest JOINT size (defects) attempted when re-solving two clusters
 * together in the merge-refinement pass.  Smaller than UF_MATCH_CAP because
 * every candidate pair pays a 2^(ki+kj) DP, and the cross-cluster gains live
 * in small adjacent clusters. */
#define UF_MERGE_CAP 12u

/* Mini-graph node cap for the correlated second pass: the shot's defects
 * plus the endpoints of every boosted edge.  A shot exceeding it keeps its
 * pass-1 answer; at sub-threshold error rates this does not happen. */
#define UF_PASS2_MAXT 128u

/* ------------------------------------------------------------------ */
/*  Graph                                                              */
/* ------------------------------------------------------------------ */
struct moonlab_uf_decoder {
    size_t    ndet;         /* detector nodes; nodes >= ndet are virtual boundary */
    size_t    nnode;        /* ndet + one boundary node per boundary edge */
    size_t    nobs;
    size_t    nedge;
    uint32_t* ea;           /* endpoints, ea[e] < nnode */
    uint32_t* eb;
    uint32_t* elen;         /* quantised growth length, >= 1 */
    double*   ewt;          /* real edge weight (log-likelihood) for matching */
    uint64_t* eobs;
    /* CSR adjacency over nodes */
    size_t*   adj_off;      /* nnode + 1 */
    uint32_t* adj_edge;     /* 2 * nedge */
    /* All-pairs shortest paths between detectors, and to the boundary,
     * precomputed once.  A cluster is resolved by EXACT minimum-weight
     * matching of its defects using these distances -- the peel produces a
     * valid correction but not a minimum-weight one, which is the whole
     * accuracy gap to blossom.  dist_to[i*ndet+j] is the shortest weighted
     * path i->j and obs_to the observables it flips; bdist[j]/bobs[j] are the
     * same to the nearest boundary. */
    double*   dist_to;      /* ndet * ndet, or NULL if not built */
    uint64_t* obs_to;       /* ndet * ndet */
    double*   bdist;        /* ndet */
    uint64_t* bobs;         /* ndet */
    /* Log path multiplicity: ln of the number of minimum-weight paths
     * realising each shortest distance.  MWPM scores a pairing by its best
     * path alone, but a pairing with N equally cheap paths is N times as
     * probable: -ln P = w - ln N.  Feeding the matcher w - ln N instead of w
     * is the maximum-likelihood score within the pairing approximation, and
     * is accuracy MWPM leaves on the table. */
    double*   lm_to;        /* ndet * ndet */
    double*   lm_b;         /* ndet */
    int       have_apsp;
    /* Correlated two-pass decoding.  corr_off/corr_dst/corr_w is a CSR of
     * DIRECTED links: for trigger edge e used by pass 1, corr_dst lists the
     * partner edges of every mechanism containing e and corr_w their
     * conditional weights ln((1-P(partner|e))/P(partner|e)), precomputed at
     * construction.  pred/bpred are the APSP predecessor traces needed to
     * recover which edges a pass-1 pairing actually used: pred[i*nnode+v]
     * is the edge into v on the stored shortest path from detector i, and
     * bpred[v] the edge into v on the path from the nearest boundary.
     * All NULL on a plain decoder. */
    size_t*   corr_off;     /* nedge + 1, or NULL */
    uint32_t* corr_dst;
    double*   corr_w;
    uint32_t* pred;         /* ndet * nnode */
    uint32_t* bpred;        /* nnode */
};

/* Per-thread scratch.  Sized once; cleared between shots through dirty
 * lists so a shot costs its own defect neighbourhood, not the whole graph. */
typedef struct {
    uint32_t* parent;       /* union-find over nodes */
    uint8_t*  parity;       /* per root: odd number of defects */
    uint8_t*  touches_b;    /* per root: cluster contains the boundary node */
    uint32_t* next;         /* intrusive member list, per node */
    uint32_t* tail;         /* per root: last member */
    uint32_t* grown;        /* per edge */
    uint8_t*  full;         /* per edge: fully grown */
    uint8_t*  syn;          /* per node: current syndrome bit */
    uint8_t*  innode;       /* per node: touched this shot */
    uint32_t* dirty_node;
    size_t    ndirty_node;
    uint32_t* dirty_edge;
    size_t    ndirty_edge;
    uint32_t* odd_roots;
    uint32_t* stack;        /* BFS order / peel stack */
    uint32_t* tree_edge;    /* per node: edge to its BFS parent */
    uint32_t* tree_par;     /* per node: BFS parent */
    uint8_t*  visited;
    /* Spanning forest of the grown region, collected during growth: an edge
     * enters it only when it actually merged two distinct clusters.  Peeling
     * over every closed edge instead would walk a graph with cycles and pick
     * an arbitrary path through each cluster; the merge edges are a forest by
     * construction and follow the order growth reached them, which is the
     * cheapest route the growth process found. */
    uint32_t* mhead;        /* per node: first incident merge-edge slot */
    uint32_t* mnext;        /* per slot: next slot at the same node */
    uint32_t* mslot_edge;   /* per slot: edge id */
    size_t    nslot;
    uint8_t*  tile;         /* UF_TILE x ndet, shot-major syndrome scratch */
    /* Exact per-cluster matching scratch. */
    uint32_t* defs;         /* lit detectors this shot */
    uint8_t*  used;         /* per gathered defect: already in a group */
    uint32_t* grp;          /* joint-solve concat buffer */
    double*   dp;           /* subset-DP cost table, 1<<UF_MATCH_CAP */
    uint32_t* dpchoice;     /* subset-DP backtrack */
    uint32_t* cl_off;       /* cluster starts into cl_defs, ncl+1 */
    uint32_t* cl_defs;      /* defects grouped by cluster */
    double*   cl_cost;      /* per cluster: -ln P of its solution (+1 spare) */
    uint64_t* cl_obs;       /* per cluster: observable mask (+1 spare) */
    uint32_t* cl_next;      /* merge chains over original clusters */
    uint32_t* cl_size;      /* group size at representative; 0 if absorbed */
    uint32_t* cl_np;        /* per cluster: pairings captured at its solve */
    double*   cl_cost0;     /* per cluster: pass-1 solo cost (correlated) */
    uint64_t* cl_obs0;      /* per cluster: pass-1 solo observables */
    uint8_t*  cl_flag;      /* per group: contains a flagged defect */
    uint32_t* mlog_i;       /* pass-1 accepted merges, in order */
    uint32_t* mlog_j;
    double*   mlog_c;       /* their joint costs */
    uint64_t* mlog_o;       /* their joint observables */
    size_t    nmlog;
    /* Frontier bookkeeping so a growth round can advance straight to the
     * next edge closure instead of creeping one unit at a time. */
    uint32_t* fstamp;       /* per edge: round id it last joined the frontier */
    uint32_t* fsides;       /* per edge: growing endpoints this round (1 or 2) */
    uint32_t* flist;        /* frontier edge list for this round */
    uint32_t  round_id;
    /* Correlated pass-2 scratch; allocated only for a correlated decoder. */
    uint32_t* pairs;        /* pass-1 pairings, (u, v|UINT32_MAX) per pairing */
    uint32_t* mpairs;       /* pairings of one re-solved merged group */
    uint32_t* ulist;        /* edges used by pass 1 this shot */
    uint32_t* estamp;       /* per edge: shot stamp when used */
    uint32_t* ostamp;       /* per edge: shot stamp when boosted */
    double*   oval;         /* per edge: boosted weight, valid under ostamp */
    uint32_t* olist;        /* boosted edges this shot */
    uint32_t* ov_sa;        /* per boosted edge slot in olist: endpoint slots */
    uint32_t* ov_sb;        /*   in the mini-graph (UINT32_MAX = boundary)   */
    uint32_t  ecur;         /* shot stamp for estamp/ostamp */
    uint32_t* tnod;         /* mini-graph slot -> detector id, UF_PASS2_MAXT */
    double*   ddist;        /* dynamic pairwise tables, UF_PASS2_MAXT^2 */
    uint64_t* dobsm;
    double*   dlm;
    double*   dbd;          /* dynamic boundary tables, UF_PASS2_MAXT */
    uint64_t* dbo;
    double*   dblm;
    double*   wdist;        /* relaxation work arrays, UF_PASS2_MAXT + 1 */
    uint64_t* wobs;
    double*   wlm;
} uf_scratch;

static void uf_scratch_free(uf_scratch* s) {
    if (!s) return;
    free(s->parent); free(s->parity); free(s->touches_b); free(s->next);
    free(s->tail); free(s->grown); free(s->full); free(s->syn);
    free(s->innode); free(s->dirty_node); free(s->dirty_edge);
    free(s->odd_roots); free(s->stack); free(s->tree_edge);
    free(s->tree_par); free(s->visited);
    free(s->mhead); free(s->mnext); free(s->mslot_edge); free(s->tile);
    free(s->fstamp); free(s->fsides); free(s->flist);
    free(s->defs); free(s->used); free(s->grp); free(s->dp); free(s->dpchoice);
    free(s->cl_off); free(s->cl_defs); free(s->cl_cost); free(s->cl_obs);
    free(s->cl_next); free(s->cl_size);
    free(s->cl_np); free(s->cl_cost0); free(s->cl_obs0);
    free(s->cl_flag); free(s->mlog_i); free(s->mlog_j);
    free(s->mlog_c); free(s->mlog_o);
    free(s->pairs); free(s->mpairs);
    free(s->ulist); free(s->estamp); free(s->ostamp);
    free(s->oval); free(s->olist); free(s->ov_sa); free(s->ov_sb);
    free(s->tnod); free(s->ddist); free(s->dobsm); free(s->dlm);
    free(s->dbd); free(s->dbo); free(s->dblm);
    free(s->wdist); free(s->wobs); free(s->wlm);
    memset(s, 0, sizeof(*s));
}

static int uf_scratch_init(uf_scratch* s, size_t nnode, size_t nedge, size_t ndet,
                           int correlated) {
    memset(s, 0, sizeof(*s));
    s->parent    = (uint32_t*)malloc(nnode * sizeof(uint32_t));
    s->parity    = (uint8_t*)calloc(nnode, 1);
    s->touches_b = (uint8_t*)calloc(nnode, 1);
    s->next      = (uint32_t*)malloc(nnode * sizeof(uint32_t));
    s->tail      = (uint32_t*)malloc(nnode * sizeof(uint32_t));
    s->grown     = (uint32_t*)calloc(nedge ? nedge : 1, sizeof(uint32_t));
    s->full      = (uint8_t*)calloc(nedge ? nedge : 1, 1);
    s->syn       = (uint8_t*)calloc(nnode, 1);
    s->innode    = (uint8_t*)calloc(nnode, 1);
    s->dirty_node= (uint32_t*)malloc(nnode * sizeof(uint32_t));
    s->dirty_edge= (uint32_t*)malloc((nedge ? nedge : 1) * sizeof(uint32_t));
    s->odd_roots = (uint32_t*)malloc(nnode * sizeof(uint32_t));
    s->stack     = (uint32_t*)malloc(nnode * sizeof(uint32_t));
    s->tree_edge = (uint32_t*)malloc(nnode * sizeof(uint32_t));
    s->tree_par  = (uint32_t*)malloc(nnode * sizeof(uint32_t));
    s->visited   = (uint8_t*)calloc(nnode, 1);
    s->mhead     = (uint32_t*)malloc(nnode * sizeof(uint32_t));
    s->mnext     = (uint32_t*)malloc(2 * (nedge ? nedge : 1) * sizeof(uint32_t));
    s->mslot_edge= (uint32_t*)malloc(2 * (nedge ? nedge : 1) * sizeof(uint32_t));
    s->tile      = (uint8_t*)malloc(UF_TILE * (ndet ? ndet : 1));
    s->fstamp    = (uint32_t*)calloc(nedge ? nedge : 1, sizeof(uint32_t));
    s->fsides    = (uint32_t*)malloc((nedge ? nedge : 1) * sizeof(uint32_t));
    s->flist     = (uint32_t*)malloc((nedge ? nedge : 1) * sizeof(uint32_t));
    s->defs      = (uint32_t*)malloc((ndet ? ndet : 1) * sizeof(uint32_t));
    s->used      = (uint8_t*)malloc(ndet ? ndet : 1);
    s->grp       = (uint32_t*)malloc((ndet ? ndet : 1) * sizeof(uint32_t));
    s->dp        = (double*)malloc(((size_t)1 << UF_MATCH_CAP) * sizeof(double));
    s->dpchoice  = (uint32_t*)malloc(((size_t)1 << UF_MATCH_CAP) * sizeof(uint32_t));
    s->cl_off    = (uint32_t*)malloc(((ndet ? ndet : 1) + 1) * sizeof(uint32_t));
    s->cl_defs   = (uint32_t*)malloc((ndet ? ndet : 1) * sizeof(uint32_t));
    s->cl_cost   = (double*)malloc(((ndet ? ndet : 1) + 1) * sizeof(double));
    s->cl_obs    = (uint64_t*)malloc(((ndet ? ndet : 1) + 1) * sizeof(uint64_t));
    s->cl_next   = (uint32_t*)malloc(((ndet ? ndet : 1) + 1) * sizeof(uint32_t));
    s->cl_size   = (uint32_t*)malloc(((ndet ? ndet : 1) + 1) * sizeof(uint32_t));
    if (!s->parent || !s->parity || !s->touches_b || !s->next || !s->tail ||
        !s->grown || !s->full || !s->syn || !s->innode || !s->dirty_node ||
        !s->dirty_edge || !s->odd_roots || !s->stack || !s->tree_edge ||
        !s->tree_par || !s->visited || !s->mhead || !s->mnext ||
        !s->mslot_edge || !s->tile || !s->fstamp || !s->fsides ||
        !s->flist || !s->defs || !s->used || !s->grp || !s->dp ||
        !s->dpchoice || !s->cl_off || !s->cl_defs || !s->cl_cost ||
        !s->cl_obs || !s->cl_next || !s->cl_size) {
        uf_scratch_free(s);
        return -1;
    }
    if (correlated) {
        const size_t ne = nedge ? nedge : 1;
        const size_t T  = UF_PASS2_MAXT;
        s->pairs  = (uint32_t*)malloc(2 * ((ndet ? ndet : 1) + 1) * sizeof(uint32_t));
        s->mpairs = (uint32_t*)malloc(2 * UF_MATCH_CAP * sizeof(uint32_t));
        s->cl_np  = (uint32_t*)malloc(((ndet ? ndet : 1) + 1) * sizeof(uint32_t));
        s->cl_cost0 = (double*)malloc(((ndet ? ndet : 1) + 1) * sizeof(double));
        s->cl_obs0  = (uint64_t*)malloc(((ndet ? ndet : 1) + 1) * sizeof(uint64_t));
        s->cl_flag  = (uint8_t*)malloc((ndet ? ndet : 1) + 1);
        s->mlog_i = (uint32_t*)malloc(((ndet ? ndet : 1) + 1) * sizeof(uint32_t));
        s->mlog_j = (uint32_t*)malloc(((ndet ? ndet : 1) + 1) * sizeof(uint32_t));
        s->mlog_c = (double*)malloc(((ndet ? ndet : 1) + 1) * sizeof(double));
        s->mlog_o = (uint64_t*)malloc(((ndet ? ndet : 1) + 1) * sizeof(uint64_t));
        s->ulist  = (uint32_t*)malloc(ne * sizeof(uint32_t));
        s->estamp = (uint32_t*)calloc(ne, sizeof(uint32_t));
        s->ostamp = (uint32_t*)calloc(ne, sizeof(uint32_t));
        s->oval   = (double*)malloc(ne * sizeof(double));
        s->olist  = (uint32_t*)malloc(ne * sizeof(uint32_t));
        s->ov_sa  = (uint32_t*)malloc(ne * sizeof(uint32_t));
        s->ov_sb  = (uint32_t*)malloc(ne * sizeof(uint32_t));
        s->tnod   = (uint32_t*)malloc(T * sizeof(uint32_t));
        s->ddist  = (double*)malloc(T * T * sizeof(double));
        s->dobsm  = (uint64_t*)malloc(T * T * sizeof(uint64_t));
        s->dlm    = (double*)malloc(T * T * sizeof(double));
        s->dbd    = (double*)malloc(T * sizeof(double));
        s->dbo    = (uint64_t*)malloc(T * sizeof(uint64_t));
        s->dblm   = (double*)malloc(T * sizeof(double));
        s->wdist  = (double*)malloc((T + 1) * sizeof(double));
        s->wobs   = (uint64_t*)malloc((T + 1) * sizeof(uint64_t));
        s->wlm    = (double*)malloc((T + 1) * sizeof(double));
        if (!s->pairs || !s->mpairs || !s->cl_np || !s->cl_cost0 ||
            !s->cl_obs0 || !s->cl_flag || !s->mlog_i || !s->mlog_j ||
            !s->mlog_c || !s->mlog_o ||
            !s->ulist || !s->estamp || !s->ostamp || !s->oval ||
            !s->olist || !s->ov_sa || !s->ov_sb || !s->tnod || !s->ddist ||
            !s->dobsm || !s->dlm || !s->dbd || !s->dbo || !s->dblm ||
            !s->wdist || !s->wobs || !s->wlm) {
            uf_scratch_free(s);
            return -1;
        }
    }
    return 0;
}

/* Distance context for the exact cluster matcher: either the static APSP
 * tables indexed by global detector id (pass 1), or per-shot dynamic tables
 * indexed by mini-graph slot (pass 2).  All scores are -ln P within the
 * pairing approximation: weight minus log path multiplicity. */
typedef struct {
    const double*   dist;   /* pairwise distance, row stride `stride` */
    const uint64_t* obsm;   /* observables along the stored shortest path */
    const double*   lm;     /* log multiplicity of minimum-weight paths */
    const double*   bd;     /* distance to the nearest boundary, per row */
    const uint64_t* bo;
    const double*   blm;
    size_t          stride;
} uf_dctx;

static uint64_t uf_match_cluster(const uf_dctx* c,
                                 const uint32_t* ids, unsigned k,
                                 double* dp, uint32_t* choice, double* out_cost,
                                 uint32_t* pairs_out, unsigned* npairs_out);

static uint32_t uf_find(uint32_t* parent, uint32_t x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];     /* path halving */
        x = parent[x];
    }
    return x;
}

/* Bring a node into this shot's working set on first touch. */
static inline void uf_touch(uf_scratch* s, uint32_t v, uint32_t ndet) {
    if (s->innode[v]) return;
    s->innode[v] = 1;
    s->parent[v] = v;
    s->parity[v] = 0;
    /* Nodes at or past ndet are virtual boundary nodes: a cluster reaching
     * one has somewhere to discharge odd parity and stops growing. */
    s->touches_b[v] = (uint8_t)(v >= ndet);
    s->next[v] = UINT32_MAX;
    s->tail[v] = v;
    s->visited[v] = 0;
    s->mhead[v] = UINT32_MAX;
    s->dirty_node[s->ndirty_node++] = v;
}

static void uf_union(uf_scratch* s, uint32_t ra, uint32_t rb) {
    if (ra == rb) return;
    /* splice member lists, combine parity and boundary contact */
    s->next[s->tail[ra]] = rb;
    s->tail[ra] = s->tail[rb];
    s->parity[ra] = (uint8_t)(s->parity[ra] ^ s->parity[rb]);
    s->touches_b[ra] = (uint8_t)(s->touches_b[ra] | s->touches_b[rb]);
    s->parent[rb] = ra;
}

/* Solve every cluster of the current shot exactly and then run the merge
 * refinement, all against the distance context `c`.  `ids` maps each
 * cl_defs entry to its row in the context tables (global detector ids for
 * the static pass, mini-graph slots for the correlated pass); NULL means
 * the entry index itself.  Group chains (cl_next/cl_size) are rebuilt, so
 * the surviving-group structure is valid on return.
 *
 * When `pairs` is non-NULL, every cluster's INITIAL pairings are captured
 * into pairs[2*cl_off[cl] ..] with the count in s->cl_np[cl] -- the
 * correlated pass reads them back for unmerged groups, so only groups the
 * refinement actually merged pay a re-solve for their pairings.  The solo
 * cost and observables are stashed in cl_cost0/cl_obs0 at the same time.
 *
 * When `flags` is non-NULL (the correlated second pass), a cluster none of
 * whose defect slots is flagged has provably static rows, so its solo
 * solution is read back from cl_cost0/cl_obs0 instead of re-running the
 * DP. */
static uint64_t uf_solve_clusters(const uf_dctx* c, uf_scratch* s, unsigned ncl,
                                  const uint32_t* ids, uint32_t* pairs,
                                  const uint8_t* flags) {
    if (flags) {
        for (unsigned cl = 0; cl < ncl; cl++) {
            uint8_t fl = 0;
            for (uint32_t t = s->cl_off[cl]; t < s->cl_off[cl + 1]; t++)
                if (flags[t]) { fl = 1; break; }
            s->cl_flag[cl] = fl;
        }
    }
    for (unsigned cl = 0; cl < ncl; cl++) {
        if (flags && !s->cl_flag[cl]) {
            s->cl_cost[cl] = s->cl_cost0[cl];
            s->cl_obs[cl] = s->cl_obs0[cl];
            continue;
        }
        unsigned g = 0;
        for (uint32_t t = s->cl_off[cl]; t < s->cl_off[cl + 1]; t++)
            s->grp[g++] = ids ? ids[t] : t;
        unsigned np = 0;
        s->cl_obs[cl] = uf_match_cluster(c, s->grp, g, s->dp, s->dpchoice,
                                         &s->cl_cost[cl],
                                         pairs ? pairs + 2 * s->cl_off[cl] : NULL,
                                         pairs ? &np : NULL);
        if (pairs) {
            s->cl_np[cl] = np;
            s->cl_cost0[cl] = s->cl_cost[cl];
            s->cl_obs0[cl] = s->cl_obs[cl];
        }
    }
    /* Merge refinement: the cluster partition forbids pairings that
     * cross clusters, which is the one restriction blossom does not
     * share and where it wins near threshold -- growth stops a
     * cluster the moment it turns even, so two even clusters can sit
     * adjacent with cross pairings cheaper than their internal ones.
     * Greedily re-solve pairs of clusters jointly and keep any merge
     * that lowers the total -ln P; each accepted merge strictly
     * improves the global score, so this converges. */
    /* Groups are chains over the ORIGINAL clusters -- defect data is
     * never moved.  Group state (size, cost, obs) lives at the
     * representative; an absorbed cluster has cl_size[c] == 0. */
    for (unsigned cl = 0; cl < ncl; cl++) {
        s->cl_next[cl] = UINT32_MAX;
        s->cl_size[cl] = s->cl_off[cl + 1] - s->cl_off[cl];
    }
    if (pairs) s->nmlog = 0;
    if (flags) {
        /* Replay pass 1's accepted merges so the unflagged sector keeps its
         * pass-1 structure without re-evaluation; only groups a boost
         * touches pay a joint re-solve.  Weights only decrease in pass 2,
         * so a merge pass 1 accepted stays beneficial; even if a boost
         * shifted the balance, the group's DP still returns that group's
         * exact optimum, so the correction stays valid. */
        for (size_t m = 0; m < s->nmlog; m++) {
            unsigned bi = s->mlog_i[m], bj = s->mlog_j[m];
            double jc; uint64_t jo;
            if (s->cl_flag[bi] || s->cl_flag[bj]) {
                unsigned g = 0;
                for (unsigned q = bi; q != UINT32_MAX; q = s->cl_next[q])
                    for (uint32_t t = s->cl_off[q]; t < s->cl_off[q + 1]; t++)
                        s->grp[g++] = ids ? ids[t] : t;
                for (unsigned q = bj; q != UINT32_MAX; q = s->cl_next[q])
                    for (uint32_t t = s->cl_off[q]; t < s->cl_off[q + 1]; t++)
                        s->grp[g++] = ids ? ids[t] : t;
                jo = uf_match_cluster(c, s->grp, g, s->dp, s->dpchoice, &jc,
                                      NULL, NULL);
            } else {
                jc = s->mlog_c[m]; jo = s->mlog_o[m];
            }
            unsigned tail = bi;
            while (s->cl_next[tail] != UINT32_MAX) tail = s->cl_next[tail];
            s->cl_next[tail] = bj;
            s->cl_size[bi] += s->cl_size[bj];
            s->cl_size[bj] = 0;
            s->cl_cost[bi] = jc;
            s->cl_obs[bi] = jo;
            s->cl_flag[bi] = (uint8_t)(s->cl_flag[bi] | s->cl_flag[bj]);
        }
    }
    for (int improved = 1; improved && ncl > 1;) {
        improved = 0;
        double best_gain = 1e-9;
        unsigned bi = 0, bj = 0;
        for (unsigned i = 0; i + 1 < ncl; i++) {
            if (!s->cl_size[i]) continue;
            for (unsigned j = i + 1; j < ncl; j++) {
                if (!s->cl_size[j]) continue;
                /* In the correlated second pass only pairs touching a
                 * boosted group can gain anything new; the rest were
                 * already settled by the replayed pass-1 refinement. */
                if (flags && !s->cl_flag[i] && !s->cl_flag[j]) continue;
                unsigned kt = s->cl_size[i] + s->cl_size[j];
                if (kt > UF_MERGE_CAP) continue;
                unsigned g = 0;
                for (unsigned q = i; q != UINT32_MAX; q = s->cl_next[q])
                    for (uint32_t t = s->cl_off[q]; t < s->cl_off[q + 1]; t++)
                        s->grp[g++] = ids ? ids[t] : t;
                for (unsigned q = j; q != UINT32_MAX; q = s->cl_next[q])
                    for (uint32_t t = s->cl_off[q]; t < s->cl_off[q + 1]; t++)
                        s->grp[g++] = ids ? ids[t] : t;
                double jc;
                uint64_t jo = uf_match_cluster(c, s->grp, g,
                                               s->dp, s->dpchoice, &jc,
                                               NULL, NULL);
                double gain = s->cl_cost[i] + s->cl_cost[j] - jc;
                if (gain > best_gain) {
                    best_gain = gain; bi = i; bj = j;
                    /* stash the winning joint solution in the spare
                     * slot past the last cluster */
                    s->cl_obs[ncl] = jo;
                    s->cl_cost[ncl] = jc;
                }
            }
        }
        if (best_gain > 1e-9) {
            if (pairs) {
                /* log the accepted merge for the correlated replay */
                s->mlog_i[s->nmlog] = bi;
                s->mlog_j[s->nmlog] = bj;
                s->mlog_c[s->nmlog] = s->cl_cost[ncl];
                s->mlog_o[s->nmlog] = s->cl_obs[ncl];
                s->nmlog++;
            }
            /* attach bj's chain to the tail of bi's */
            unsigned tail = bi;
            while (s->cl_next[tail] != UINT32_MAX) tail = s->cl_next[tail];
            s->cl_next[tail] = bj;
            s->cl_size[bi] += s->cl_size[bj];
            s->cl_size[bj] = 0;
            s->cl_cost[bi] = s->cl_cost[ncl];
            s->cl_obs[bi] = s->cl_obs[ncl];
            if (flags)
                s->cl_flag[bi] = (uint8_t)(s->cl_flag[bi] | s->cl_flag[bj]);
            improved = 1;
        }
    }
    uint64_t obs = 0;
    for (unsigned cl = 0; cl < ncl; cl++)
        if (s->cl_size[cl]) obs ^= s->cl_obs[cl];
    return obs;
}

/* Correlated second pass.  Recovers which edges pass 1's pairings used by
 * walking the APSP predecessor traces, applies the precomputed conditional
 * reweighting of every used edge's mechanism partners, computes exact
 * boosted distances between the shot's defects, and re-solves the pass-1
 * cluster partition under them.  Returns 1 with *obs_out set when pass 2
 * produced an answer; 0 keeps the pass-1 correction (no boost applied, or
 * the shot exceeded the mini-graph cap).
 *
 * Boosted distances are exact, not approximate: lowering some edge weights
 * means any new shortest path decomposes into static shortest segments
 * between boosted-edge endpoints, joined by the boosted edges themselves
 * (virtual boundary nodes have degree 1, so no path passes through the
 * boundary).  Dijkstra on the mini-graph over {defects} + {boosted
 * endpoints} + {boundary sink}, whose edge set is the static APSP distances
 * plus the boosted edges, therefore reproduces the true boosted distance,
 * observable flips, and path log-multiplicity. */
static int uf_pass2(const moonlab_uf_decoder_t* d, uf_scratch* s,
                    const uf_dctx* st, unsigned ncl, uint64_t* obs_out) {
    const size_t nd = d->ndet;
    const size_t nn = d->nnode;
    const size_t T  = UF_PASS2_MAXT;

    /* --- recover the pass-1 pairings of every surviving group and mark
     * the edges their stored paths run through.  Unmerged groups reuse the
     * pairings captured at their initial solve; only groups the refinement
     * merged pay a re-solve. ------------------------------------------- */
    s->ecur++;
    size_t nused = 0;
    for (unsigned cl = 0; cl < ncl; cl++) {
        if (!s->cl_size[cl]) continue;
        const uint32_t* gp_pairs;
        unsigned gnp;
        if (s->cl_next[cl] == UINT32_MAX) {
            gp_pairs = s->pairs + 2 * s->cl_off[cl];
            gnp = s->cl_np[cl];
        } else {
            unsigned g = 0;
            for (unsigned q = cl; q != UINT32_MAX; q = s->cl_next[q])
                for (uint32_t t = s->cl_off[q]; t < s->cl_off[q + 1]; t++)
                    s->grp[g++] = s->cl_defs[t];
            uf_match_cluster(st, s->grp, g, s->dp, s->dpchoice, NULL,
                             s->mpairs, &gnp);
            gp_pairs = s->mpairs;
        }
        for (unsigned i = 0; i < gnp; i++) {
            uint32_t u = gp_pairs[2 * i], v = gp_pairs[2 * i + 1];
            if (v == UINT32_MAX) {
                uint32_t x = u; size_t guard = 0;
                while (x < nd && guard++ < nn) {
                    uint32_t e = d->bpred[x];
                    if (e == UINT32_MAX) break;
                    if (s->estamp[e] != s->ecur) {
                        s->estamp[e] = s->ecur; s->ulist[nused++] = e;
                    }
                    x = (d->ea[e] == x) ? d->eb[e] : d->ea[e];
                }
            } else {
                const uint32_t* pr = d->pred + (size_t)u * nn;
                uint32_t x = v; size_t guard = 0;
                while (x != u && guard++ < nn) {
                    uint32_t e = pr[x];
                    if (e == UINT32_MAX) break;
                    if (s->estamp[e] != s->ecur) {
                        s->estamp[e] = s->ecur; s->ulist[nused++] = e;
                    }
                    x = (d->ea[e] == x) ? d->eb[e] : d->ea[e];
                }
            }
        }
    }

    /* --- conditional reweighting of the used edges' partners ----------- */
    size_t nov = 0;
    for (size_t i = 0; i < nused; i++) {
        uint32_t e = s->ulist[i];
        for (size_t k = d->corr_off[e]; k < d->corr_off[e + 1]; k++) {
            uint32_t f = d->corr_dst[k];
            double w = d->corr_w[k];
            if (w >= d->ewt[f]) continue;          /* no gain */
            if (s->ostamp[f] != s->ecur) {
                s->ostamp[f] = s->ecur; s->oval[f] = w; s->olist[nov++] = f;
            } else if (w < s->oval[f]) {
                s->oval[f] = w;                     /* strongest trigger wins */
            }
        }
    }
    if (nov == 0) return 0;

    /* --- which boosts matter, and which defects do they touch? --------- */
    /* A boosted edge changes a matching decision only if it improves some
     * defect-pair or defect-boundary distance.  Test each boosted edge
     * against the static tables -- path i -> endpoint, cross, endpoint -> j
     * (or boundary) -- keeping only the edges that improve something and
     * flagging the defects whose rows they improve.  Most pass-1
     * corrections boost partners whose region holds no defects, so most
     * shots keep nothing and skip the whole second pass; when it does run,
     * only the flagged defects need a mini-graph Dijkstra, the rest keep
     * their static rows.  An improving pair flags BOTH endpoints (each
     * orientation of the test flags its entry side), so an unflagged
     * defect provably has every pairwise and boundary distance unchanged.
     * Reroutes that need two boosts where neither improves anything alone
     * are the one thing this drops; single-boost reroutes, the entire
     * first-order effect, are kept exactly. */
    const size_t ndef = s->cl_off[ncl];
    if (ndef > T) return 0;
    for (size_t i = 0; i < ndef; i++) s->used[i] = 0;   /* affected flags */
    {
        size_t nkeep = 0;
        for (size_t o = 0; o < nov; o++) {
            uint32_t f = s->olist[o];
            const double w = s->oval[f];
            int keep = 0;
            for (int orient = 0; orient < 2; orient++) {
                uint32_t a = orient ? d->eb[f] : d->ea[f];
                uint32_t b = orient ? d->ea[f] : d->eb[f];
                if (a >= nd) continue;    /* cannot enter through the boundary */
                const double* rowa = d->dist_to + (size_t)a * nd;
                const double* rowb = (b < nd) ? d->dist_to + (size_t)b * nd : NULL;
                for (size_t ii = 0; ii < ndef; ii++) {
                    uint32_t di = s->cl_defs[ii];
                    double head = rowa[di] + w;
                    if (!rowb) {          /* boosted boundary edge */
                        if (head < d->bdist[di] * (1.0 - 1e-9)) {
                            keep = 1; s->used[ii] = 1;
                        }
                        continue;
                    }
                    /* Triangle prune: d(i,j) <= d(i,b) + d(b,j) and
                     * bdist(i) <= d(i,b) + bdist(b), so improving EITHER
                     * through entry a needs d(i,a) + w < d(i,b).  One load
                     * rejects this defect for this edge orientation. */
                    if (head >= rowb[di]) continue;
                    if (head + d->bdist[b] < d->bdist[di] * (1.0 - 1e-9)) {
                        keep = 1; s->used[ii] = 1;
                        continue;
                    }
                    const double* rowi = d->dist_to + (size_t)di * nd;
                    for (size_t jj = 0; jj < ndef; jj++) {
                        if (jj == ii) continue;
                        uint32_t dj = s->cl_defs[jj];
                        if (head + rowb[dj] < rowi[dj] * (1.0 - 1e-9)) {
                            keep = 1; s->used[ii] = 1;
                            break;
                        }
                    }
                }
            }
            if (keep) s->olist[nkeep++] = f;
        }
        nov = nkeep;
    }
    if (nov == 0) return 0;

    /* --- mini-graph: shot defects, then boosted interior endpoints ----- */
    size_t nt = ndef;
    for (size_t i = 0; i < ndef; i++) s->tnod[i] = s->cl_defs[i];
    for (size_t o = 0; o < nov; o++) {
        uint32_t f = s->olist[o];
        uint32_t ends[2]; ends[0] = d->ea[f]; ends[1] = d->eb[f];
        uint32_t slot[2];
        for (int side = 0; side < 2; side++) {
            uint32_t x = ends[side];
            if (x >= nd) { slot[side] = UINT32_MAX; continue; }   /* boundary */
            size_t j = 0;
            while (j < nt && s->tnod[j] != x) j++;
            if (j == nt) {
                if (nt >= T) return 0;
                s->tnod[nt++] = x;
            }
            slot[side] = (uint32_t)j;
        }
        if (slot[0] == UINT32_MAX && slot[1] == UINT32_MAX) {
            slot[0] = 0;    /* degenerate boundary-boundary edge: inert */
            s->oval[f] = 1e300;
        }
        s->ov_sa[o] = slot[0];
        s->ov_sb[o] = slot[1];
    }

    /* --- exact boosted distances by shortcut relaxation ----------------- */
    /* Any boosted shortest path decomposes at its LAST boosted crossing:
     * prefix (source to the entry endpoint, itself possibly boosted), the
     * boosted edge, then a purely static tail (exit endpoint to target) --
     * virtual boundary nodes have degree 1, so no path passes through the
     * boundary.  Iterating that relaxation to fixpoint over the kept
     * boosted edges therefore reproduces the exact boosted distance for
     * arbitrarily chained crossings, with the chain length bounding the
     * round count.  Path log-multiplicity rides along on strict
     * improvements; tie accumulation is not attempted here (the static
     * segments' multiplicities, the dominant term, come from the tables). */
    const size_t bnd = nt;                          /* boundary sink slot */
    for (size_t src = 0; src < ndef; src++) {
        if (!s->used[src]) {
            /* nothing involving this defect improved: its rows ARE the
             * static tables */
            const uint32_t gi = s->cl_defs[src];
            const double*   rd = d->dist_to + (size_t)gi * nd;
            const uint64_t* ro = d->obs_to + (size_t)gi * nd;
            const double*   rl = d->lm_to + (size_t)gi * nd;
            for (size_t j = 0; j < ndef; j++) {
                const uint32_t gj = s->cl_defs[j];
                s->ddist[src * T + j] = rd[gj];
                s->dobsm[src * T + j] = ro[gj];
                s->dlm[src * T + j]   = rl[gj];
            }
            s->dbd[src] = d->bdist[gi];
            s->dbo[src] = d->bobs[gi];
            s->dblm[src] = d->lm_b[gi];
            continue;
        }
        {
            const uint32_t gi = s->cl_defs[src];
            const double*   rd = d->dist_to + (size_t)gi * nd;
            const uint64_t* ro = d->obs_to + (size_t)gi * nd;
            const double*   rl = d->lm_to + (size_t)gi * nd;
            for (size_t j = 0; j < nt; j++) {
                const uint32_t gj = s->tnod[j];
                s->wdist[j] = rd[gj]; s->wobs[j] = ro[gj]; s->wlm[j] = rl[gj];
            }
            s->wdist[bnd] = d->bdist[gi];
            s->wobs[bnd] = d->bobs[gi];
            s->wlm[bnd] = d->lm_b[gi];
        }
        for (int round = 0; round < 8; round++) {
            int changed = 0;
            for (size_t o = 0; o < nov; o++) {
                const uint32_t f = s->olist[o];
                const double w = s->oval[f];
                for (int orient = 0; orient < 2; orient++) {
                    const uint32_t sl_in  = orient ? s->ov_sb[o] : s->ov_sa[o];
                    const uint32_t sl_out = orient ? s->ov_sa[o] : s->ov_sb[o];
                    if (sl_in == UINT32_MAX) continue;   /* no entry via boundary */
                    const double base = s->wdist[sl_in] + w;
                    if (base >= 1e300) continue;
                    const uint64_t eobs_in = s->wobs[sl_in] ^ d->eobs[f];
                    const double lm_in = s->wlm[sl_in];
                    if (sl_out == UINT32_MAX) {          /* exit straight to bnd */
                        if (base < s->wdist[bnd] * (1.0 - 1e-12) - 1e-12) {
                            s->wdist[bnd] = base;
                            s->wobs[bnd] = eobs_in;
                            s->wlm[bnd] = lm_in;
                            changed = 1;
                        }
                        continue;
                    }
                    const uint32_t gout = s->tnod[sl_out];
                    const double*   trd = d->dist_to + (size_t)gout * nd;
                    const uint64_t* tro = d->obs_to + (size_t)gout * nd;
                    const double*   trl = d->lm_to + (size_t)gout * nd;
                    for (size_t y = 0; y <= nt; y++) {
                        double tail, tlm; uint64_t tobs;
                        if (y == bnd) {
                            tail = d->bdist[gout]; tobs = d->bobs[gout];
                            tlm = d->lm_b[gout];
                        } else {
                            const uint32_t gy = s->tnod[y];
                            tail = trd[gy]; tobs = tro[gy]; tlm = trl[gy];
                        }
                        const double cand = base + tail;
                        if (cand < s->wdist[y] * (1.0 - 1e-12) - 1e-12) {
                            s->wdist[y] = cand;
                            s->wobs[y] = eobs_in ^ tobs;
                            s->wlm[y] = lm_in + tlm;
                            changed = 1;
                        }
                    }
                }
            }
            if (!changed) break;
        }
        for (size_t j = 0; j < ndef; j++) {
            s->ddist[src * T + j] = s->wdist[j];
            s->dobsm[src * T + j] = s->wobs[j];
            s->dlm[src * T + j]   = s->wlm[j];
        }
        s->dbd[src] = s->wdist[bnd];
        s->dbo[src] = s->wobs[bnd];
        s->dblm[src] = s->wlm[bnd];
    }

    /* --- re-match the pass-1 partition under the boosted distances ----- */
    uf_dctx dy;
    dy.dist = s->ddist; dy.obsm = s->dobsm; dy.lm = s->dlm;
    dy.bd = s->dbd; dy.bo = s->dbo; dy.blm = s->dblm;
    dy.stride = T;
    *obs_out = uf_solve_clusters(&dy, s, ncl, NULL, NULL, s->used);
    return 1;
}

/* Decode one shot.  Returns the observable mask to apply. */
static uint64_t uf_decode_shot(const moonlab_uf_decoder_t* d, uf_scratch* s,
                               const uint8_t* syn_row) {
    const size_t ndet = d->ndet;

    /* --- reset only what the previous shot touched ------------------- */
    for (size_t i = 0; i < s->ndirty_node; i++) {
        uint32_t v = s->dirty_node[i];
        s->innode[v] = 0; s->syn[v] = 0; s->visited[v] = 0;
    }
    for (size_t i = 0; i < s->ndirty_edge; i++) {
        uint32_t e = s->dirty_edge[i];
        s->grown[e] = 0; s->full[e] = 0;
    }
    s->ndirty_node = 0;
    s->ndirty_edge = 0;
    s->nslot = 0;

    /* --- seed clusters at lit detectors ------------------------------ */
    size_t nodd = 0;
    for (size_t i = 0; i < ndet; i++) {
        if (syn_row[i]) {
            uint32_t v = (uint32_t)i;
            uf_touch(s, v, (uint32_t)ndet);
            s->syn[v] = 1;
            s->parity[v] = 1;
            s->odd_roots[nodd++] = v;
        }
    }
    if (nodd == 0) return 0;

    

    /* --- grow odd clusters ------------------------------------------- */
    for (;;) {
        /* Collect the current odd cluster roots. */
        size_t live = 0;
        for (size_t i = 0; i < nodd; i++) {
            uint32_t r = uf_find(s->parent, s->odd_roots[i]);
            if (s->parity[r] && !s->touches_b[r]) {
                int seen = 0;
                for (size_t j = 0; j < live; j++)
                    if (s->odd_roots[j] == r) { seen = 1; break; }
                if (!seen) s->odd_roots[live++] = r;
            }
        }
        nodd = live;
        if (nodd == 0) break;

        /* Collect this round's frontier: every not-yet-closed edge incident
         * to a growing cluster, with how many of its endpoints are growing. */
        size_t nfront = 0;
        s->round_id++;
        for (size_t i = 0; i < nodd; i++) {
            uint32_t root = s->odd_roots[i];
            for (uint32_t v = root; v != UINT32_MAX; v = s->next[v]) {
                for (size_t k = d->adj_off[v]; k < d->adj_off[v + 1]; k++) {
                    uint32_t e = d->adj_edge[k];
                    if (s->full[e]) continue;
                    if (s->fstamp[e] != s->round_id) {
                        s->fstamp[e] = s->round_id;
                        s->fsides[e] = 1;
                        s->flist[nfront++] = e;
                        if (s->grown[e] == 0) s->dirty_edge[s->ndirty_edge++] = e;
                    } else {
                        s->fsides[e]++;
                    }
                }
            }
        }
        if (nfront == 0) break;

        /* Advance every frontier edge by the largest amount that closes at
         * least one of them.  Creeping one unit per round instead makes the
         * round count scale with the weight quantisation, so each growth
         * round re-walks every cluster member for nothing. */
        uint32_t delta = UINT32_MAX;
        for (size_t f = 0; f < nfront; f++) {
            uint32_t e = s->flist[f];
            uint32_t need = d->elen[e] - s->grown[e];
            uint32_t sides = s->fsides[e];
            uint32_t steps = (need + sides - 1u) / sides;
            if (steps < delta) delta = steps;
        }
        if (delta == 0) delta = 1;

        for (size_t f = 0; f < nfront; f++) {
            uint32_t e = s->flist[f];
            s->grown[e] += delta * s->fsides[e];
            if (s->grown[e] >= d->elen[e]) {
                s->full[e] = 1;
                uint32_t a = d->ea[e], b = d->eb[e];
                uf_touch(s, a, (uint32_t)ndet); uf_touch(s, b, (uint32_t)ndet);
                uint32_t ra = uf_find(s->parent, a);
                uint32_t rb = uf_find(s->parent, b);
                if (ra != rb) {
                    /* this edge joins two clusters: a forest edge */
                    uint32_t s0 = (uint32_t)s->nslot++;
                    s->mslot_edge[s0] = e;
                    s->mnext[s0] = s->mhead[a];
                    s->mhead[a] = s0;
                    uint32_t s1 = (uint32_t)s->nslot++;
                    s->mslot_edge[s1] = e;
                    s->mnext[s1] = s->mhead[b];
                    s->mhead[b] = s1;
                    uf_union(s, ra, rb);
                }
            }
        }
    }

    uint64_t obs = 0;

    /* --- exact resolution: minimum-weight matching per cluster -------- */
    if (d->have_apsp) {
        /* Gather this shot's defects. */
        size_t ndef = 0;
        for (size_t i = 0; i < s->ndirty_node; i++) {
            uint32_t v = s->dirty_node[i];
            if (v < ndet && s->syn[v]) s->defs[ndef++] = v;
        }
        int ok = 1;
        for (size_t i = 0; i < ndef; i++) s->used[i] = 0;
        /* Collect clusters into grouped storage (ndef is small, so the
         * O(ndef) scan per cluster is cheap in aggregate). */
        unsigned ncl = 0;
        uint32_t fill = 0;
        for (size_t a = 0; a < ndef && ok; a++) {
            if (s->used[a]) continue;
            uint32_t root = uf_find(s->parent, s->defs[a]);
            s->cl_off[ncl] = fill;
            for (size_t b = a; b < ndef; b++) {
                if (s->used[b]) continue;
                if (uf_find(s->parent, s->defs[b]) != root) continue;
                s->used[b] = 1;
                s->cl_defs[fill++] = s->defs[b];
            }
            if (fill - s->cl_off[ncl] > UF_MATCH_CAP) { ok = 0; break; }
            ncl++;
        }
        if (ok) {
            s->cl_off[ncl] = fill;
            uf_dctx st;
            st.dist = d->dist_to; st.obsm = d->obs_to; st.lm = d->lm_to;
            st.bd = d->bdist; st.bo = d->bobs; st.blm = d->lm_b;
            st.stride = ndet;
            const int correlated = (d->corr_off && d->pred);
            obs = uf_solve_clusters(&st, s, ncl, s->cl_defs,
                                    correlated ? s->pairs : NULL, NULL);
            /* Correlated decoder: run the second pass off the pass-1
             * correction; keep pass 1's answer when nothing was boosted. */
            if (correlated) {
                uint64_t obs2;
                if (uf_pass2(d, s, &st, ncl, &obs2)) obs = obs2;
            }
            return obs;
        }
        obs = 0;   /* fall through to the peel */
    }

    /* --- fallback: peel the grown forest ----------------------------- */
    /* The boundary must be visited first so that every tree containing it is
     * ROOTED at it.  Peeling pushes each unmatched defect up to its parent,
     * so a cluster that reached the boundary can only discharge its odd
     * parity there if the boundary sits at the root; rooting such a tree
     * anywhere else strands the parity at an interior root and emits a
     * correction that does not match the syndrome. */
    for (size_t pass = 0; pass < 2; pass++) {
    for (size_t i0 = 0; i0 < s->ndirty_node; i0++) {
        uint32_t start = s->dirty_node[i0];
        /* Pass 0 roots every tree that contains a boundary node AT that
         * node, so parity can flow to it; pass 1 handles the rest. */
        if ((pass == 0) != (start >= ndet)) continue;
        if (s->visited[start]) continue;
        /* BFS over fully grown edges from `start`. */
        size_t head = 0, tailn = 0;
        s->stack[tailn++] = start;
        s->visited[start] = 1;
        s->tree_edge[start] = UINT32_MAX;
        s->tree_par[start] = UINT32_MAX;
        while (head < tailn) {
            uint32_t v = s->stack[head++];
            for (uint32_t sl = s->mhead[v]; sl != UINT32_MAX; sl = s->mnext[sl]) {
                uint32_t e = s->mslot_edge[sl];
                uint32_t w = (d->ea[e] == v) ? d->eb[e] : d->ea[e];
                if (s->visited[w]) continue;
                s->visited[w] = 1;
                s->tree_edge[w] = e;
                s->tree_par[w] = v;
                s->stack[tailn++] = w;
            }
        }
        /* Peel from the leaves: walk the BFS order backwards so every node
         * is handled after its descendants.  A lit node pushes its defect
         * up to its parent through the tree edge, which joins the
         * correction; the boundary node absorbs whatever reaches it. */
        for (size_t j = tailn; j-- > 0;) {
            uint32_t v = s->stack[j];
            if (v >= ndet) continue;
            if (!s->syn[v]) continue;
            uint32_t e = s->tree_edge[v];
            if (e == UINT32_MAX) continue;      /* isolated: nothing to do */
            obs ^= d->eobs[e];
            s->syn[v] = 0;
            uint32_t p = s->tree_par[v];
            s->syn[p] = (uint8_t)(s->syn[p] ^ 1);
        }
    }
    }
    return obs;
}

/* ------------------------------------------------------------------ */
/* ------------------------------------------------------------------ */
/*  All-pairs shortest paths (Dijkstra, binary heap)                   */
/* ------------------------------------------------------------------ */
typedef struct { double key; uint32_t node; } uf_heap_item;

static void uf_heap_push(uf_heap_item* h, size_t* n, double key, uint32_t node) {
    size_t i = (*n)++;
    h[i].key = key; h[i].node = node;
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (h[p].key <= h[i].key) break;
        uf_heap_item t = h[p]; h[p] = h[i]; h[i] = t; i = p;
    }
}
static uf_heap_item uf_heap_pop(uf_heap_item* h, size_t* n) {
    uf_heap_item top = h[0];
    h[0] = h[--(*n)];
    size_t i = 0;
    for (;;) {
        size_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < *n && h[l].key < h[m].key) m = l;
        if (r < *n && h[r].key < h[m].key) m = r;
        if (m == i) break;
        uf_heap_item t = h[m]; h[m] = h[i]; h[i] = t; i = m;
    }
    return top;
}

/* ln(e^a + e^b) without overflow. */
static inline double uf_logaddexp(double a, double b) {
    double hi = a > b ? a : b, lo = a > b ? b : a;
    return hi + log1p(exp(lo - hi));
}

/* Dijkstra from `src` (a single node, or every boundary node at distance 0
 * when src == UINT32_MAX), writing the shortest distance, the XOR of edge
 * observables along one shortest path, and the LOG MULTIPLICITY of
 * minimum-weight paths into dist[] / obsm[] / lmult[] for the detector
 * nodes.  Multiplicity accumulates on ties relaxed from finalised nodes;
 * ties discovered after a node is finalised are dropped, which undercounts
 * slightly but never corrupts distances.
 *
 * When `pred` is non-NULL it receives, for every node, the edge into it on
 * the SAME shortest path whose observables were stored -- pred is updated
 * exactly when the stored obs is, so walking pred from a target back to the
 * source reproduces the path the matcher's obs mask came from.  The
 * correlated pass needs this to know which edges a pairing used. */
static void uf_dijkstra(const moonlab_uf_decoder_t* d, uint32_t src,
                        double* dist, uint64_t* obsm, double* lmult,
                        double* dscratch, uint64_t* oscratch, double* lscratch,
                        uint8_t* done, uf_heap_item* heap, uint32_t* pred) {
    const size_t nn = d->nnode;
    for (size_t v = 0; v < nn; v++) {
        dscratch[v] = 1e300; oscratch[v] = 0; lscratch[v] = 0.0; done[v] = 0;
    }
    if (pred)
        for (size_t v = 0; v < nn; v++) pred[v] = UINT32_MAX;
    size_t hn = 0;
    if (src == UINT32_MAX) {
        for (size_t v = d->ndet; v < nn; v++) {
            dscratch[v] = 0.0; uf_heap_push(heap, &hn, 0.0, (uint32_t)v);
        }
    } else {
        dscratch[src] = 0.0; uf_heap_push(heap, &hn, 0.0, src);
    }
    while (hn > 0) {
        uf_heap_item it = uf_heap_pop(heap, &hn);
        uint32_t u = it.node;
        if (done[u]) continue;
        done[u] = 1;
        for (size_t k = d->adj_off[u]; k < d->adj_off[u + 1]; k++) {
            uint32_t e = d->adj_edge[k];
            uint32_t w = (d->ea[e] == u) ? d->eb[e] : d->ea[e];
            double nd = dscratch[u] + d->ewt[e];
            double eps = 1e-9 * (1.0 + nd);
            if (nd < dscratch[w] - eps) {
                dscratch[w] = nd;
                oscratch[w] = oscratch[u] ^ d->eobs[e];
                lscratch[w] = lscratch[u];
                if (pred) pred[w] = e;
                uf_heap_push(heap, &hn, nd, w);
            } else if (!done[w] && nd <= dscratch[w] + eps) {
                /* another minimum-weight route into w */
                lscratch[w] = uf_logaddexp(lscratch[w], lscratch[u]);
            }
        }
    }
    for (size_t j = 0; j < d->ndet; j++) {
        dist[j] = dscratch[j]; obsm[j] = oscratch[j]; lmult[j] = lscratch[j];
    }
}

/* Precompute detector-to-detector and detector-to-boundary shortest paths.
 * Skipped (leaving the peel as the resolver) when the dense ndet*ndet tables
 * would be too large to be worth it. */
static void uf_build_apsp(moonlab_uf_decoder_t* d) {
    const size_t nd = d->ndet;
    if (nd == 0 || nd > 20000) return;          /* dense table cap */
    d->dist_to = (double*)malloc(nd * nd * sizeof(double));
    d->obs_to  = (uint64_t*)malloc(nd * nd * sizeof(uint64_t));
    d->bdist   = (double*)malloc(nd * sizeof(double));
    d->bobs    = (uint64_t*)malloc(nd * sizeof(uint64_t));
    d->lm_to   = (double*)malloc(nd * nd * sizeof(double));
    d->lm_b    = (double*)malloc(nd * sizeof(double));
    double*   ds = (double*)malloc(d->nnode * sizeof(double));
    uint64_t* os = (uint64_t*)malloc(d->nnode * sizeof(uint64_t));
    double*   ls = (double*)malloc(d->nnode * sizeof(double));
    uint8_t*  dn = (uint8_t*)malloc(d->nnode);
    uf_heap_item* hp = (uf_heap_item*)malloc((2 * d->nedge + d->nnode) * sizeof(uf_heap_item));
    if (!d->dist_to || !d->obs_to || !d->bdist || !d->bobs || !d->lm_to ||
        !d->lm_b || !ds || !os || !ls || !dn || !hp) {
        free(d->dist_to); free(d->obs_to); free(d->bdist); free(d->bobs);
        free(d->lm_to); free(d->lm_b);
        d->dist_to = NULL; d->obs_to = NULL; d->bdist = NULL; d->bobs = NULL;
        d->lm_to = NULL; d->lm_b = NULL;
        free(ds); free(os); free(ls); free(dn); free(hp);
        return;
    }
    /* The correlated pass needs the predecessor traces; without them the
     * decoder degrades to plain two-pass-off behaviour. */
    if (d->corr_off) {
        d->pred  = (uint32_t*)malloc(nd * d->nnode * sizeof(uint32_t));
        d->bpred = (uint32_t*)malloc(d->nnode * sizeof(uint32_t));
        if (!d->pred || !d->bpred) {
            free(d->pred); free(d->bpred);
            d->pred = NULL; d->bpred = NULL;
        }
    }
    for (size_t i = 0; i < nd; i++)
        uf_dijkstra(d, (uint32_t)i, d->dist_to + i * nd, d->obs_to + i * nd,
                    d->lm_to + i * nd, ds, os, ls, dn, hp,
                    d->pred ? d->pred + i * d->nnode : NULL);
    uf_dijkstra(d, UINT32_MAX, d->bdist, d->bobs, d->lm_b, ds, os, ls, dn, hp,
                d->bpred);
    free(ds); free(os); free(ls); free(dn); free(hp);
    d->have_apsp = 1;
}

/* Exact minimum-weight resolution of one cluster's defects by subset DP.
 *
 * dp[mask] is the least-weight way to resolve the defect set `mask`, where
 * the lowest-indexed defect in the set is either paired with another defect
 * in the set or (if the cluster reached the boundary) sent to the boundary.
 * This is exact minimum-weight perfect matching over the cluster -- the same
 * optimum blossom computes -- and is cheap because a cluster holds only a
 * handful of defects.  Returns the observable mask, accumulating it from the
 * chosen pairings' precomputed path observables.
 */
static uint64_t uf_match_cluster(const uf_dctx* c,
                                 const uint32_t* ids, unsigned k,
                                 double* dp, uint32_t* choice, double* out_cost,
                                 uint32_t* pairs_out, unsigned* npairs_out) {
    const size_t st = c->stride;
    const uint32_t full = (k >= 32) ? 0xFFFFFFFFu : ((1u << k) - 1u);
    dp[0] = 0.0; choice[0] = 0;
    for (uint32_t mask = 1; mask <= full; mask++) {
        /* lowest set bit = first unresolved defect */
        unsigned i = (unsigned)__builtin_ctz(mask);
        double best = 1e300; uint32_t bch = UINT32_MAX;
        /* Scores are -ln P within the pairing approximation: path weight
         * minus ln(number of minimum-weight paths).  MWPM uses the weight
         * alone, so between two equal-weight pairings it cannot prefer the
         * one realisable in more ways, which is the more probable one.
         *
         * The boundary is ALWAYS an option: a defect-to-boundary error path
         * is physically valid whether or not cluster growth happened to
         * reach the boundary, and blossom considers it for every defect.
         * Restricting it to boundary-touching clusters was an artificial
         * constraint the incumbent does not have. */
        {
            double cc = c->bd[ids[i]] - c->blm[ids[i]]
                      + dp[mask & ~(1u << i)];
            if (cc < best) { best = cc; bch = i; }   /* i -> boundary */
        }
        for (unsigned j = i + 1; j < k; j++) {
            if (!(mask & (1u << j))) continue;
            size_t ij = (size_t)ids[i] * st + ids[j];
            double cc = c->dist[ij] - c->lm[ij]
                      + dp[mask & ~(1u << i) & ~(1u << j)];
            if (cc < best) { best = cc; bch = (i << 8) | j; }
        }
        dp[mask] = best; choice[mask] = bch;
    }
    if (out_cost) *out_cost = dp[full];
    /* backtrack; optionally record the chosen pairings as (id, id) with
     * UINT32_MAX marking the boundary partner */
    uint64_t obs = 0;
    unsigned np = 0;
    uint32_t mask = full;
    while (mask) {
        uint32_t ch = choice[mask];
        unsigned i = (unsigned)__builtin_ctz(mask);
        if (ch == i) {                     /* boundary */
            obs ^= c->bo[ids[i]];
            if (pairs_out) {
                pairs_out[2 * np] = ids[i];
                pairs_out[2 * np + 1] = UINT32_MAX;
                np++;
            }
            mask &= ~(1u << i);
        } else {
            unsigned a = ch >> 8, b = ch & 0xFF;
            obs ^= c->obsm[(size_t)ids[a] * st + ids[b]];
            if (pairs_out) {
                pairs_out[2 * np] = ids[a];
                pairs_out[2 * np + 1] = ids[b];
                np++;
            }
            mask &= ~(1u << a) & ~(1u << b);
        }
    }
    if (npairs_out) *npairs_out = np;
    return obs;
}

/* ------------------------------------------------------------------ */
/*  Construction                                                       */
/* ------------------------------------------------------------------ */

/* Conditional weight of a partner edge given its correlated trigger edge
 * was decided flipped by pass 1.  Exact within the independent-mechanism
 * model: split each edge's sources into the joint mechanisms (probability
 * q of firing, flipping both edges) and the edge-only remainder, recovered
 * from the merged marginals by inverting the XOR combination,
 *   p = q(1-q_only) + q_only(1-q)  =>  q_only = (p - q) / (1 - 2q).
 * Given the trigger flipped, the flip came from a joint mechanism with
 * probability r = q(1-q_t) / (q(1-q_t) + q_t(1-q)); the partner is then
 * flipped iff an odd number of its remaining sources fired:
 *   P(partner | trigger) = r (1 - q_p) + (1 - r) q_p.
 * The returned weight ln((1-P)/P) is floored at a small positive value so
 * pass-2 shortest paths stay well defined even when P crosses 1/2. */
static double uf_cond_weight(double p_trig, double p_par, double q) {
    if (q > p_trig) q = p_trig;      /* a joint source is one of the sources */
    if (q > p_par) q = p_par;
    if (q > 0.499999) q = 0.499999;
    double qt = (p_trig - q) / (1.0 - 2.0 * q);
    double qp = (p_par - q) / (1.0 - 2.0 * q);
    if (qt < 0.0) qt = 0.0;
    if (qp < 0.0) qp = 0.0;
    double num = q * (1.0 - qt);
    double den = num + qt * (1.0 - q);
    double r = (den > 0.0) ? num / den : 1.0;
    double pc = r * (1.0 - qp) + (1.0 - r) * qp;
    if (pc < 1e-15) pc = 1e-15;
    if (pc > 1.0 - 1e-9) pc = 1.0 - 1e-9;
    double w = log((1.0 - pc) / pc);
    if (w < 1e-6) w = 1e-6;
    return w;
}

static moonlab_uf_decoder_t* uf_build(
    size_t num_detectors, size_t num_observables,
    const uint32_t* edge_a, const uint32_t* edge_b,
    const double* edge_weight, const uint64_t* edge_obs,
    size_t num_edges,
    const double* edge_prob,
    const uint32_t* corr_a, const uint32_t* corr_b,
    const double* corr_joint_p, size_t num_corr) {
    if (num_detectors == 0 || num_edges == 0 || !edge_a || !edge_b || !edge_obs)
        return NULL;
    if (num_observables > 64) return NULL;
    if (num_corr > 0 && (!edge_prob || !corr_a || !corr_b || !corr_joint_p))
        return NULL;

    moonlab_uf_decoder_t* d = (moonlab_uf_decoder_t*)calloc(1, sizeof(*d));
    if (!d) return NULL;
    /* Every boundary edge gets its OWN virtual boundary node.  Sharing one
     * node makes the boundary a hub: two clusters on opposite sides of the
     * code that each reach it become a single cluster, and peeling is then
     * free to route a defect across the whole patch through that hub.  The
     * damage grows with code size, which shows up as a logical error rate
     * that rises with distance instead of falling. */
    size_t nb = 0;
    for (size_t e = 0; e < num_edges; e++)
        if (edge_a[e] == MOONLAB_UF_BOUNDARY || edge_b[e] == MOONLAB_UF_BOUNDARY) nb++;
    d->ndet  = num_detectors;
    d->nnode = num_detectors + (nb ? nb : 1);
    d->nobs  = num_observables;
    d->nedge = num_edges;
    d->ea   = (uint32_t*)malloc(num_edges * sizeof(uint32_t));
    d->eb   = (uint32_t*)malloc(num_edges * sizeof(uint32_t));
    d->elen = (uint32_t*)malloc(num_edges * sizeof(uint32_t));
    d->ewt  = (double*)malloc(num_edges * sizeof(double));
    d->eobs = (uint64_t*)malloc(num_edges * sizeof(uint64_t));
    d->adj_off = (size_t*)calloc(d->nnode + 1, sizeof(size_t));
    d->adj_edge = (uint32_t*)malloc(2 * num_edges * sizeof(uint32_t));
    if (!d->ea || !d->eb || !d->elen || !d->ewt || !d->eobs ||
        !d->adj_off || !d->adj_edge) {
        moonlab_uf_decoder_free(d); return NULL;
    }

    /* Quantise weights into integer growth lengths.  Only the ratio between
     * weights matters, so scale the smallest positive weight to 1 and cap
     * the spread; an unbounded range would make growth rounds scale with
     * the weight ratio rather than with the defect count. */
    double wmin = 0.0;
    if (edge_weight) {
        for (size_t e = 0; e < num_edges; e++)
            if (edge_weight[e] > 0.0 && (wmin == 0.0 || edge_weight[e] < wmin))
                wmin = edge_weight[e];
    }
    size_t next_bnode = num_detectors;
    for (size_t e = 0; e < num_edges; e++) {
        uint32_t a = edge_a[e];
        uint32_t b = edge_b[e];
        if (a == MOONLAB_UF_BOUNDARY) a = (uint32_t)next_bnode++;
        if (b == MOONLAB_UF_BOUNDARY) b = (uint32_t)next_bnode++;
        if (a >= d->nnode || b >= d->nnode) { moonlab_uf_decoder_free(d); return NULL; }
        d->ea[e] = a;
        d->eb[e] = b;
        d->eobs[e] = edge_obs[e];
        /* Matching uses the real weight; a non-positive or missing weight
         * falls back to unit so the graph stays connected. */
        d->ewt[e] = (edge_weight && edge_weight[e] > 0.0) ? edge_weight[e] : 1.0;
        /* Quantise onto a fine scale.  Rounding w/wmin straight to an integer
         * collapses everything within 2x of the cheapest edge onto length 1,
         * so most of the graph looks equally cheap and growth stops
         * distinguishing likely from unlikely faults -- the decoder then
         * fails to convert extra code distance into error suppression.
         * MOONLAB_UF_WSCALE units per multiple of the cheapest weight keeps
         * that resolution; the cap bounds how many growth rounds the widest
         * edge can cost. */
        uint32_t len = MOONLAB_UF_WSCALE;
        if (edge_weight && wmin > 0.0 && edge_weight[e] > 0.0) {
            double r = edge_weight[e] / wmin;
            if (r > 24.0) r = 24.0;
            double q = r * (double)MOONLAB_UF_WSCALE;
            len = (uint32_t)(q + 0.5);
            if (len == 0) len = 1;
        }
        /* Growth is in HALF-edges: a cluster advances each incident edge by
         * one unit per round, so an edge between two growing clusters closes
         * in half the rounds of an edge whose far end is not growing (the
         * boundary, or a node in an already-even cluster).  Storing the
         * length as 2*weight is what encodes that.  With length 1 a single
         * increment closes any edge, every incident edge completes in the
         * same round, and a defect pair is as likely to be routed to the
         * boundary as to each other -- which decodes a two-defect syndrome
         * as two boundary corrections and flips the observable. */
        d->elen[e] = 2u * len;
        d->adj_off[a + 1]++;
        d->adj_off[b + 1]++;
    }
    for (size_t v = 0; v < d->nnode; v++) d->adj_off[v + 1] += d->adj_off[v];
    size_t* cur = (size_t*)malloc(d->nnode * sizeof(size_t));
    if (!cur) { moonlab_uf_decoder_free(d); return NULL; }
    memcpy(cur, d->adj_off, d->nnode * sizeof(size_t));
    for (size_t e = 0; e < num_edges; e++) {
        d->adj_edge[cur[d->ea[e]]++] = (uint32_t)e;
        d->adj_edge[cur[d->eb[e]]++] = (uint32_t)e;
    }
    free(cur);

    /* Correlation links: build the directed CSR of (trigger -> partner,
     * conditional weight) before the APSP so it knows to keep pred traces. */
    if (num_corr > 0) {
        for (size_t l = 0; l < num_corr; l++) {
            if (corr_a[l] >= num_edges || corr_b[l] >= num_edges ||
                corr_a[l] == corr_b[l] ||
                !(corr_joint_p[l] > 0.0) || corr_joint_p[l] >= 0.5) {
                moonlab_uf_decoder_free(d); return NULL;
            }
        }
        for (size_t e = 0; e < num_edges; e++) {
            if (!(edge_prob[e] > 0.0) || edge_prob[e] >= 1.0) {
                moonlab_uf_decoder_free(d); return NULL;
            }
        }
        d->corr_off = (size_t*)calloc(num_edges + 1, sizeof(size_t));
        d->corr_dst = (uint32_t*)malloc(2 * num_corr * sizeof(uint32_t));
        d->corr_w   = (double*)malloc(2 * num_corr * sizeof(double));
        size_t* ccur = (size_t*)malloc(num_edges * sizeof(size_t));
        if (!d->corr_off || !d->corr_dst || !d->corr_w || !ccur) {
            free(ccur); moonlab_uf_decoder_free(d); return NULL;
        }
        for (size_t l = 0; l < num_corr; l++) {
            d->corr_off[corr_a[l] + 1]++;
            d->corr_off[corr_b[l] + 1]++;
        }
        for (size_t e = 0; e < num_edges; e++)
            d->corr_off[e + 1] += d->corr_off[e];
        memcpy(ccur, d->corr_off, num_edges * sizeof(size_t));
        for (size_t l = 0; l < num_corr; l++) {
            uint32_t u = corr_a[l], v = corr_b[l];
            double q = corr_joint_p[l];
            size_t su = ccur[u]++;
            d->corr_dst[su] = v;
            d->corr_w[su] = uf_cond_weight(edge_prob[u], edge_prob[v], q);
            size_t sv = ccur[v]++;
            d->corr_dst[sv] = u;
            d->corr_w[sv] = uf_cond_weight(edge_prob[v], edge_prob[u], q);
        }
        free(ccur);
    }

    uf_build_apsp(d);   /* enables exact cluster matching; peel is the fallback */
    return d;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */
moonlab_uf_decoder_t* moonlab_uf_decoder_new(
    size_t num_detectors, size_t num_observables,
    const uint32_t* edge_a, const uint32_t* edge_b,
    const double* edge_weight, const uint64_t* edge_obs,
    size_t num_edges) {
    return uf_build(num_detectors, num_observables, edge_a, edge_b,
                    edge_weight, edge_obs, num_edges,
                    NULL, NULL, NULL, NULL, 0);
}

moonlab_uf_decoder_t* moonlab_uf_decoder_new_correlated(
    size_t num_detectors, size_t num_observables,
    const uint32_t* edge_a, const uint32_t* edge_b,
    const double* edge_weight, const uint64_t* edge_obs,
    size_t num_edges,
    const double* edge_prob,
    const uint32_t* corr_a, const uint32_t* corr_b,
    const double* corr_joint_p, size_t num_corr) {
    return uf_build(num_detectors, num_observables, edge_a, edge_b,
                    edge_weight, edge_obs, num_edges,
                    edge_prob, corr_a, corr_b, corr_joint_p, num_corr);
}

void moonlab_uf_decoder_free(moonlab_uf_decoder_t* d) {
    if (!d) return;
    free(d->ea); free(d->eb); free(d->elen); free(d->ewt); free(d->eobs);
    free(d->adj_off); free(d->adj_edge);
    free(d->dist_to); free(d->obs_to); free(d->bdist); free(d->bobs);
    free(d->lm_to); free(d->lm_b);
    free(d->corr_off); free(d->corr_dst); free(d->corr_w);
    free(d->pred); free(d->bpred);
    free(d);
}

size_t moonlab_uf_decoder_num_edges(const moonlab_uf_decoder_t* d) {
    return d ? d->nedge : 0;
}

long moonlab_uf_decode_batch(moonlab_uf_decoder_t* d, const uint8_t* det,
                             size_t num_shots, int num_threads, uint8_t* obs_out) {
    if (!d || !det || !obs_out || num_shots == 0) return -1;

    int nthreads = num_threads;
#ifdef _OPENMP
    if (nthreads <= 0) nthreads = omp_get_max_threads();
#else
    if (nthreads <= 0) nthreads = 1;
#endif
    if ((size_t)nthreads > num_shots) nthreads = (int)num_shots;
    if (nthreads < 1) nthreads = 1;

    const size_t base = num_shots / (size_t)nthreads;
    const size_t rem  = num_shots % (size_t)nthreads;
    int err = 0;

#ifdef _OPENMP
#   pragma omp parallel for num_threads(nthreads) schedule(static, 1) reduction(|:err)
#endif
    for (int tid = 0; tid < nthreads; tid++) {
        size_t bs = base + ((size_t)tid < rem ? 1 : 0);
        size_t off = (size_t)tid * base + ((size_t)tid < rem ? (size_t)tid : rem);
        if (bs == 0) continue;
        uf_scratch s;
        if (uf_scratch_init(&s, d->nnode, d->nedge, d->ndet,
                            d->corr_off != NULL) != 0) { err |= 1; continue; }
        const size_t ndet = d->ndet;
        for (size_t t0 = 0; t0 < bs; t0 += UF_TILE) {
            const size_t tn = (bs - t0 < UF_TILE) ? bs - t0 : UF_TILE;
            /* Transpose a tile of shots into shot-major order.  The source
             * read runs along `shot` for a fixed detector, which is
             * contiguous; decoding then reads one shot's detectors
             * contiguously.  Reading det[i * num_shots + shot] directly per
             * shot instead strides by num_shots for every detector, which is
             * a cache miss per detector per shot. */
            for (size_t i = 0; i < ndet; i++) {
                const uint8_t* src = det + i * num_shots + off + t0;
                uint8_t* dst = s.tile + i;
                for (size_t k = 0; k < tn; k++) dst[k * ndet] = src[k];
            }
            for (size_t k = 0; k < tn; k++) {
                const size_t shot = off + t0 + k;
                uint64_t obs = uf_decode_shot(d, &s, s.tile + k * ndet);
                for (size_t o = 0; o < d->nobs; o++)
                    obs_out[o * num_shots + shot] = (uint8_t)((obs >> o) & 1u);
            }
        }
        uf_scratch_free(&s);
    }
    return err ? -1 : (long)num_shots;
}
