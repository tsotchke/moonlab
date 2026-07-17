#!/usr/bin/env python3
"""Cross-binding differential: the Python (ctypes) dense binding vs the numpy
reference oracle pinned in the corpus.

This reproduces every corpus circuit through ``moonlab.core.QuantumState`` and
checks the full probability vector and <Z_i>/<Z_iZ_j> expectations against the
reference within 1e-10. It catches binding-marshalling bugs a C-only test cannot
see: endianness, row/column order, qubit-index conventions, angle sign.

Runs two ways:
  * pytest  -- ``pytest tests/differential/test_diff_python.py``
  * direct  -- ``python3 tests/differential/test_diff_python.py [corpus.json]``
    prints ``DIFF_PY_RESULT ...`` and exits 0 (pass) / 1 (fail) / 77 (skip).

Skips CLEANLY (exit 77, logged reason) when the moonlab Python binding or its
native library is unavailable, rather than failing -- but the run always reports
how many cases were actually exercised so an empty run cannot masquerade as
green. Set MOONLAB_LIB_DIR so the binding finds libquantumsim.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

TOL = 1e-10

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_corpus() -> Path:
    env = os.environ.get("MOONLAB_DIFF_CORPUS")
    if env:
        return Path(env)
    return _REPO_ROOT / "build" / "differential" / "corpus.json"


def _find_binding():
    """Import moonlab, adding the in-repo binding to sys.path if needed.
    Returns (module, None) or (None, reason)."""
    try:
        import numpy  # noqa: F401
    except Exception as e:  # pragma: no cover
        return None, f"numpy unavailable: {e}"
    binding = _REPO_ROOT / "bindings" / "python"
    if binding.is_dir() and str(binding) not in sys.path:
        sys.path.insert(0, str(binding))
    try:
        from moonlab.core import QuantumState  # noqa: F401
    except Exception as e:
        return None, f"moonlab.core import failed ({type(e).__name__}: {e})"
    return QuantumState, None


def _apply(state, g) -> None:
    name = g["name"]
    q = g["qubits"]
    theta = g.get("angle", 0.0)
    if name == "h":
        state.h(q[0])
    elif name == "x":
        state.x(q[0])
    elif name == "y":
        state.y(q[0])
    elif name == "z":
        state.z(q[0])
    elif name == "s":
        state.s(q[0])
    elif name == "sdg":
        state.sdg(q[0])
    elif name == "t":
        state.t(q[0])
    elif name == "tdg":
        state.tdg(q[0])
    elif name == "rx":
        state.rx(q[0], theta)
    elif name == "ry":
        state.ry(q[0], theta)
    elif name == "rz":
        state.rz(q[0], theta)
    elif name == "p":
        state.phase(q[0], theta)
    elif name == "cx":
        state.cnot(q[0], q[1])
    elif name == "cz":
        state.cz(q[0], q[1])
    elif name == "swap":
        state.swap(q[0], q[1])
    elif name == "cp":
        state.cphase(q[0], q[1], theta)
    elif name == "ccx":
        state.toffoli(q[0], q[1], q[2])
    else:
        raise ValueError(f"unknown gate {name}")


def _exp_z(probs, n):
    out = []
    for qq in range(n):
        e = 0.0
        for i in range(len(probs)):
            e += -probs[i] if (i >> qq) & 1 else probs[i]
        out.append(e)
    return out


def _exp_zz(probs, pairs):
    out = []
    for (a, b) in pairs:
        e = 0.0
        for i in range(len(probs)):
            sa = -1 if (i >> a) & 1 else 1
            sb = -1 if (i >> b) & 1 else 1
            e += sa * sb * probs[i]
        out.append(e)
    return out


def run(corpus_path: Path):
    """Returns dict(status, exercised, failed, skipped, reason, failures)."""
    QuantumState, reason = _find_binding()
    if QuantumState is None:
        return {"status": "SKIP", "exercised": 0, "failed": 0, "reason": reason,
                "failures": []}
    if not corpus_path.is_file():
        return {"status": "SKIP", "exercised": 0, "failed": 0,
                "reason": f"corpus not found: {corpus_path}", "failures": []}

    corpus = json.loads(corpus_path.read_text())
    failures = []
    exercised = 0
    for case in corpus["cases"]:
        n = case["num_qubits"]
        ref = case["reference"]
        st = QuantumState(n)
        for g in case["gates"]:
            _apply(st, g)
        probs = st.probabilities()
        exercised += 1
        # full probability vector
        dev = max(abs(float(probs[i]) - ref["probabilities"][i]) for i in range(len(probs)))
        if dev > TOL:
            failures.append((case["id"], case["seed"], "prob", dev))
            continue
        # expectations
        pairs = [(a, b) for (a, b, _v) in ref["exp_zz"]]
        ez = _exp_z(probs, n)
        devz = max((abs(ez[q] - ref["exp_z"][q]) for q in range(n)), default=0.0)
        ezz = _exp_zz(probs, pairs)
        devzz = max((abs(ezz[k] - ref["exp_zz"][k][2]) for k in range(len(pairs))), default=0.0)
        if devz > TOL:
            failures.append((case["id"], case["seed"], "expZ", devz))
        if devzz > TOL:
            failures.append((case["id"], case["seed"], "expZZ", devzz))

    status = "FAIL" if failures else "PASS"
    return {"status": status, "exercised": exercised, "failed": len(failures),
            "reason": "ok", "failures": failures}


# --------------------------------------------------------------------------
# pytest entry point
# --------------------------------------------------------------------------
def test_cross_binding_python():
    import pytest
    res = run(_default_corpus())
    if res["status"] == "SKIP":
        pytest.skip(res["reason"])
    assert res["exercised"] > 0, "no cases exercised (silent-empty run)"
    if res["failures"]:
        msg = "\n".join(f"{cid} seed={seed} {what} dev={dev:.3e}"
                        for (cid, seed, what, dev) in res["failures"][:20])
        assert not res["failures"], f"Python binding diverged from reference:\n{msg}"


def main() -> int:
    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_corpus()
    res = run(corpus)
    for (cid, seed, what, dev) in res["failures"][:50]:
        print(f"  FAIL  case={cid} seed={seed} {what} [python] vs reference "
              f"dev={dev:.3e} (>{TOL:.0e})", file=sys.stderr)
    print(f'DIFF_PY_RESULT status={res["status"]} cases={res["exercised"]} '
          f'failed={res["failed"]} reason="{res["reason"]}"')
    if res["status"] == "SKIP":
        return 77
    return 0 if res["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
