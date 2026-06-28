"""moonlab <-> pymatching subprocess bridge.

Invoked by the decoder-bench dispatcher when the
`MOONLAB_DECODER_PYMATCHING` slot is selected.  Reads a JSON
syndrome record on stdin, writes the correction byte vector as
hex to stdout.

Protocol:

  stdin:
    { "distance": 5,
      "is_toric": true,
      "syndromes": [0, 1, 0, 0, 1, ...],
      "rng_seed": 12345 }

  stdout:
    OK <hex correction byte vector>
  -- or --
    ERR <message>

Requires:  pip install pymatching numpy
"""

from __future__ import annotations

import json
import sys


def _build_toric_matching_graph(d):
    """Build pymatching.Matching for a d x d toric code (Z stabilisers).

    Moonlab's edge-index convention (matches `decoder_bench.c`):

      - Horizontal edge h(x, y) at index ``x*d + y`` connects vertex
        ``(x, y)`` to vertex ``(x+1, y)`` -- runs in the +X direction.
      - Vertical edge v(x, y) at index ``d*d + x*d + y`` connects
        vertex ``(x, y)`` to vertex ``(x, y+1)`` -- runs in the +Y
        direction.

    Vertex scalar index is ``x*d + y`` (row-major, same as the
    syndrome buffer).  Fault ids are assigned so that ids ``[0, d*d)``
    correspond to moonlab's horizontal edges and ids ``[d*d, 2*d*d)``
    to vertical edges, matching the correction-byte layout the C
    dispatcher expects.

    All-1 weights -- minimum-weight matching collapses to
    minimum-weight by edge count, which on a uniform torus equals
    the L1 geodesic moonlab's in-tree MWPM_EXACT computes.

    An earlier version of this builder had the H/V loops swapped,
    so pymatching's "horizontal" fault ids encoded +Y steps and its
    "vertical" ones encoded +X.  After C->python round-trip the
    corrections were transposed across the lattice diagonal,
    inflating logical-error rates roughly 20x at d=5 p=0.01 (same
    shape as the pre-fix geodesic bug in `torus_edge_between`).
    """
    import pymatching
    m = pymatching.Matching()
    # n_vertices = d * d  (implicit -- pymatching grows as edges are added).
    fault_id = 0
    # Horizontal edges first (fault_ids 0..d*d-1).  Edge h(x, y) connects
    # vertex (x, y) to vertex ((x+1)%d, y); steps in +X.
    for x in range(d):
        for y in range(d):
            u = x * d + y
            v = ((x + 1) % d) * d + y
            m.add_edge(u, v, fault_ids={fault_id}, weight=1.0)
            fault_id += 1
    # Vertical edges (fault_ids d*d..2*d*d-1).  Edge v(x, y) connects
    # vertex (x, y) to vertex (x, (y+1)%d); steps in +Y.
    for x in range(d):
        for y in range(d):
            u = x * d + y
            v = x * d + ((y + 1) % d)
            m.add_edge(u, v, fault_ids={fault_id}, weight=1.0)
            fault_id += 1
    return m, fault_id


def main():
    raw = sys.stdin.read()
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"ERR json: {e}", flush=True)
        return 1

    if not req.get("is_toric"):
        print("ERR only_toric_supported", flush=True)
        return 1
    d = int(req["distance"])
    syndromes = req["syndromes"]
    if len(syndromes) != d * d:
        print(f"ERR syndrome_length_{len(syndromes)}_vs_{d*d}", flush=True)
        return 1

    try:
        import pymatching  # noqa: F401
        import numpy as np
    except ImportError as e:
        print(f"ERR import: pip install pymatching numpy ({e})", flush=True)
        return 1

    m, n_edges = _build_toric_matching_graph(d)
    s = np.array(syndromes, dtype=np.uint8)

    try:
        correction = m.decode(s)
    except Exception as e:
        print(f"ERR decode: {e}", flush=True)
        return 1

    # correction is a numpy array of int over [fault_ids]; truncate /
    # pad to n_edges and emit as hex of length 2 * d * d bytes.
    out_bytes = bytes(int(c) & 1 for c in correction[:n_edges])
    print("OK " + out_bytes.hex(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
