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
    /* Frontier bookkeeping so a growth round can advance straight to the
     * next edge closure instead of creeping one unit at a time. */
    uint32_t* fstamp;       /* per edge: round id it last joined the frontier */
    uint32_t* fsides;       /* per edge: growing endpoints this round (1 or 2) */
    uint32_t* flist;        /* frontier edge list for this round */
    uint32_t  round_id;
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
    memset(s, 0, sizeof(*s));
}

static int uf_scratch_init(uf_scratch* s, size_t nnode, size_t nedge, size_t ndet) {
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
    return 0;
}

static uint64_t uf_match_cluster(const moonlab_uf_decoder_t* d,
                                 const uint32_t* def, unsigned k,
                                 double* dp, uint32_t* choice, double* out_cost);

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
            for (unsigned c = 0; c < ncl; c++)
                s->cl_obs[c] = uf_match_cluster(
                    d, s->cl_defs + s->cl_off[c], s->cl_off[c + 1] - s->cl_off[c],
                    s->dp, s->dpchoice, &s->cl_cost[c]);

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
            for (unsigned c = 0; c < ncl; c++) {
                s->cl_next[c] = UINT32_MAX;
                s->cl_size[c] = s->cl_off[c + 1] - s->cl_off[c];
            }
            for (int improved = 1; improved && ncl > 1;) {
                improved = 0;
                double best_gain = 1e-9;
                unsigned bi = 0, bj = 0;
                for (unsigned i = 0; i + 1 < ncl; i++) {
                    if (!s->cl_size[i]) continue;
                    for (unsigned j = i + 1; j < ncl; j++) {
                        if (!s->cl_size[j]) continue;
                        unsigned kt = s->cl_size[i] + s->cl_size[j];
                        if (kt > UF_MERGE_CAP) continue;
                        unsigned g = 0;
                        for (unsigned c = i; c != UINT32_MAX; c = s->cl_next[c])
                            for (uint32_t t = s->cl_off[c]; t < s->cl_off[c + 1]; t++)
                                s->grp[g++] = s->cl_defs[t];
                        for (unsigned c = j; c != UINT32_MAX; c = s->cl_next[c])
                            for (uint32_t t = s->cl_off[c]; t < s->cl_off[c + 1]; t++)
                                s->grp[g++] = s->cl_defs[t];
                        double jc;
                        uint64_t jo = uf_match_cluster(d, s->grp, g,
                                                       s->dp, s->dpchoice, &jc);
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
                    /* attach bj's chain to the tail of bi's */
                    unsigned tail = bi;
                    while (s->cl_next[tail] != UINT32_MAX) tail = s->cl_next[tail];
                    s->cl_next[tail] = bj;
                    s->cl_size[bi] += s->cl_size[bj];
                    s->cl_size[bj] = 0;
                    s->cl_cost[bi] = s->cl_cost[ncl];
                    s->cl_obs[bi] = s->cl_obs[ncl];
                    improved = 1;
                }
            }
            for (unsigned c = 0; c < ncl; c++)
                if (s->cl_size[c]) obs ^= s->cl_obs[c];
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
 * slightly but never corrupts distances. */
static void uf_dijkstra(const moonlab_uf_decoder_t* d, uint32_t src,
                        double* dist, uint64_t* obsm, double* lmult,
                        double* dscratch, uint64_t* oscratch, double* lscratch,
                        uint8_t* done, uf_heap_item* heap) {
    const size_t nn = d->nnode;
    for (size_t v = 0; v < nn; v++) {
        dscratch[v] = 1e300; oscratch[v] = 0; lscratch[v] = 0.0; done[v] = 0;
    }
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
    for (size_t i = 0; i < nd; i++)
        uf_dijkstra(d, (uint32_t)i, d->dist_to + i * nd, d->obs_to + i * nd,
                    d->lm_to + i * nd, ds, os, ls, dn, hp);
    uf_dijkstra(d, UINT32_MAX, d->bdist, d->bobs, d->lm_b, ds, os, ls, dn, hp);
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
static uint64_t uf_match_cluster(const moonlab_uf_decoder_t* d,
                                 const uint32_t* def, unsigned k,
                                 double* dp, uint32_t* choice, double* out_cost) {
    const size_t nd = d->ndet;
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
            double c = d->bdist[def[i]] - d->lm_b[def[i]]
                     + dp[mask & ~(1u << i)];
            if (c < best) { best = c; bch = i; }   /* i -> boundary */
        }
        for (unsigned j = i + 1; j < k; j++) {
            if (!(mask & (1u << j))) continue;
            size_t ij = (size_t)def[i] * nd + def[j];
            double c = d->dist_to[ij] - d->lm_to[ij]
                     + dp[mask & ~(1u << i) & ~(1u << j)];
            if (c < best) { best = c; bch = (i << 8) | j; }
        }
        dp[mask] = best; choice[mask] = bch;
    }
    if (out_cost) *out_cost = dp[full];
    /* backtrack */
    uint64_t obs = 0;
    uint32_t mask = full;
    while (mask) {
        uint32_t ch = choice[mask];
        unsigned i = (unsigned)__builtin_ctz(mask);
        if (ch == i) {                     /* boundary */
            obs ^= d->bobs[def[i]];
            mask &= ~(1u << i);
        } else {
            unsigned a = ch >> 8, b = ch & 0xFF;
            obs ^= d->obs_to[def[a] * nd + def[b]];
            mask &= ~(1u << a) & ~(1u << b);
        }
    }
    return obs;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */
moonlab_uf_decoder_t* moonlab_uf_decoder_new(
    size_t num_detectors, size_t num_observables,
    const uint32_t* edge_a, const uint32_t* edge_b,
    const double* edge_weight, const uint64_t* edge_obs,
    size_t num_edges) {
    if (num_detectors == 0 || num_edges == 0 || !edge_a || !edge_b || !edge_obs)
        return NULL;
    if (num_observables > 64) return NULL;

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

    uf_build_apsp(d);   /* enables exact cluster matching; peel is the fallback */
    return d;
}

void moonlab_uf_decoder_free(moonlab_uf_decoder_t* d) {
    if (!d) return;
    free(d->ea); free(d->eb); free(d->elen); free(d->ewt); free(d->eobs);
    free(d->adj_off); free(d->adj_edge);
    free(d->dist_to); free(d->obs_to); free(d->bdist); free(d->bobs);
    free(d->lm_to); free(d->lm_b);
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
        if (uf_scratch_init(&s, d->nnode, d->nedge, d->ndet) != 0) { err |= 1; continue; }
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
