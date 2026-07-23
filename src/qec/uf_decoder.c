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

#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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
    uint64_t* eobs;
    /* CSR adjacency over nodes */
    size_t*   adj_off;      /* nnode + 1 */
    uint32_t* adj_edge;     /* 2 * nedge */
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
} uf_scratch;

static void uf_scratch_free(uf_scratch* s) {
    if (!s) return;
    free(s->parent); free(s->parity); free(s->touches_b); free(s->next);
    free(s->tail); free(s->grown); free(s->full); free(s->syn);
    free(s->innode); free(s->dirty_node); free(s->dirty_edge);
    free(s->odd_roots); free(s->stack); free(s->tree_edge);
    free(s->tree_par); free(s->visited);
    memset(s, 0, sizeof(*s));
}

static int uf_scratch_init(uf_scratch* s, size_t nnode, size_t nedge) {
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
    if (!s->parent || !s->parity || !s->touches_b || !s->next || !s->tail ||
        !s->grown || !s->full || !s->syn || !s->innode || !s->dirty_node ||
        !s->dirty_edge || !s->odd_roots || !s->stack || !s->tree_edge ||
        !s->tree_par || !s->visited) {
        uf_scratch_free(s);
        return -1;
    }
    return 0;
}

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
                               const uint8_t* det, size_t num_shots, size_t shot) {
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

    /* --- seed clusters at lit detectors ------------------------------ */
    size_t nodd = 0;
    for (size_t i = 0; i < ndet; i++) {
        if (det[i * num_shots + shot]) {
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

        /* Grow each odd cluster by one unit along all incident edges. */
        for (size_t i = 0; i < nodd; i++) {
            uint32_t root = s->odd_roots[i];
            for (uint32_t v = root; v != UINT32_MAX; v = s->next[v]) {
                for (size_t k = d->adj_off[v]; k < d->adj_off[v + 1]; k++) {
                    uint32_t e = d->adj_edge[k];
                    if (s->full[e]) continue;
                    if (s->grown[e] == 0) s->dirty_edge[s->ndirty_edge++] = e;
                    s->grown[e]++;
                    if (s->grown[e] >= d->elen[e]) {
                        s->full[e] = 1;
                        uint32_t a = d->ea[e], b = d->eb[e];
                        uf_touch(s, a, (uint32_t)ndet); uf_touch(s, b, (uint32_t)ndet);
                        uf_union(s, uf_find(s->parent, a), uf_find(s->parent, b));
                    }
                }
            }
        }
    }

    /* --- peel the grown forest --------------------------------------- */
    uint64_t obs = 0;
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
            for (size_t k = d->adj_off[v]; k < d->adj_off[v + 1]; k++) {
                uint32_t e = d->adj_edge[k];
                if (!s->full[e]) continue;
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
    d->eobs = (uint64_t*)malloc(num_edges * sizeof(uint64_t));
    d->adj_off = (size_t*)calloc(d->nnode + 1, sizeof(size_t));
    d->adj_edge = (uint32_t*)malloc(2 * num_edges * sizeof(uint32_t));
    if (!d->ea || !d->eb || !d->elen || !d->eobs || !d->adj_off || !d->adj_edge) {
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
        uint32_t len = 1;
        if (edge_weight && wmin > 0.0 && edge_weight[e] > 0.0) {
            double r = edge_weight[e] / wmin;
            if (r > 32.0) r = 32.0;
            len = (uint32_t)(r + 0.5);
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
    return d;
}

void moonlab_uf_decoder_free(moonlab_uf_decoder_t* d) {
    if (!d) return;
    free(d->ea); free(d->eb); free(d->elen); free(d->eobs);
    free(d->adj_off); free(d->adj_edge);
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
        if (uf_scratch_init(&s, d->nnode, d->nedge) != 0) { err |= 1; continue; }
        for (size_t k = 0; k < bs; k++) {
            const size_t shot = off + k;
            uint64_t obs = uf_decode_shot(d, &s, det, num_shots, shot);
            for (size_t o = 0; o < d->nobs; o++)
                obs_out[o * num_shots + shot] = (uint8_t)((obs >> o) & 1u);
        }
        uf_scratch_free(&s);
    }
    return err ? -1 : (long)num_shots;
}
