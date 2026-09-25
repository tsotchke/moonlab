/**
 * @file  decoder_shootout_toric.h
 * @brief Toric-code lattice helpers shared by decoder_shootout.c and
 *        its unit test: edge indexing, the Z-vertex syndrome, and the
 *        logical-failure check on a residual X-pattern.
 *
 * Toric d x d, num_qubits = 2*d*d.  Horizontal edge h(x, y) at index
 * x*d+y joins vertex (x, y) to (x+1, y); vertical edge v(x, y) at
 * d*d+x*d+y joins (x, y) to (x, y+1).  This is the convention of
 * torus_edge_between in src/applications/decoder_bench.c.
 */
#ifndef MOONLAB_BENCH_DECODER_SHOOTOUT_TORIC_H
#define MOONLAB_BENCH_DECODER_SHOOTOUT_TORIC_H

static inline int h_idx(int d, int x, int y) { return x * d + y; }
static inline int v_idx(int d, int x, int y) { return d * d + x * d + y; }

/* Parity of the X-pattern on the four edges incident to vertex
 * (vx, vy): h(vx-1, vy) + h(vx, vy) + v(vx, vy-1) + v(vx, vy). */
static inline int vertex_parity(int d, const unsigned char *x_err,
                                int vx, int vy) {
    int p = 0;
    p ^= x_err[h_idx(d, (vx + d - 1) % d, vy)];
    p ^= x_err[h_idx(d, vx, vy)];
    p ^= x_err[v_idx(d, vx, (vy + d - 1) % d)];
    p ^= x_err[v_idx(d, vx, vy)];
    return p & 1;
}

/* Compute the Z-vertex syndrome from the X-error pattern. */
static inline void compute_syndrome(int d, const unsigned char *x_err,
                                     unsigned char *syndromes) {
    for (int vx = 0; vx < d; vx++)
        for (int vy = 0; vy < d; vy++)
            syndromes[vx * d + vy] = (unsigned char)vertex_parity(d, x_err, vx, vy);
}

/* A residual X-pattern r is a logical failure iff it leaves a syndrome
 * or it is a cycle in a nontrivial homology class.  For a cycle, the
 * class is read by the two logical-Z operators, which must commute
 * with every X-plaquette {h(a,b), h(a,b+1), v(a,b), v(a+1,b)}:
 *
 *   Z_1 on {h(0, y) : y}   (the horizontal edges crossing x = 1/2)
 *   Z_2 on {v(x, 0) : x}   (the vertical edges crossing y = 1/2)
 *
 * Each meets every plaquette in 0 or 2 edges, and meets the X_1 loop
 * {h(x, y0) : x} (resp. the X_2 loop {v(x0, y) : y}) exactly once.
 * {h(x, 0) : x} is the X_1 loop itself, not a Z operator: it meets
 * the plaquettes in rows y = 0 and y = d-1 once each, so parity on it
 * miscounts stabilizer residuals as failures. */
static inline int logical_failure(int d, const unsigned char *x_err) {
    for (int vx = 0; vx < d; vx++)
        for (int vy = 0; vy < d; vy++)
            if (vertex_parity(d, x_err, vx, vy)) return 1;
    int z1 = 0, z2 = 0;
    for (int k = 0; k < d; k++) {
        z1 ^= x_err[h_idx(d, 0, k)];
        z2 ^= x_err[v_idx(d, k, 0)];
    }
    return (z1 | z2) & 1;
}

#endif /* MOONLAB_BENCH_DECODER_SHOOTOUT_TORIC_H */
