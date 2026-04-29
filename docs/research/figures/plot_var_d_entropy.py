#!/usr/bin/env python3
"""
Plot the headline figure for the variational-D paper.

Reads `benchmarks/results/ca_mps_var_d_vs_plain_dmrg_2026-04-29.json`
and generates two panels:

  Left:  bar chart of S(|psi>) (plain DMRG) vs S(|phi>) (var-D)
         at each (n, g) point, log-y axis.
  Right: log-scale entropy reduction ratio S_psi / S_phi vs g,
         color-coded by n.

Produces `figs/var_d_entropy_reduction.png` (and .pdf for the paper).

Usage:  python3 plot_var_d_entropy.py [path/to/json]

Dependencies: matplotlib (pip), no external data fetches.
"""
import json
import os
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
except ImportError:
    sys.stderr.write("This script needs matplotlib (pip install matplotlib).\n")
    sys.exit(2)


def load(json_path: Path) -> list[dict]:
    with open(json_path) as f:
        d = json.load(f)
    return d["points"]


def make_figure(points, out_basename: str = "var_d_entropy_reduction"):
    # Group by n
    n_groups: dict[int, list[dict]] = {}
    for p in points:
        n_groups.setdefault(p["n"], []).append(p)
    for n in n_groups:
        n_groups[n].sort(key=lambda p: p["g"])

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left panel: bar chart of S_psi vs S_phi
    bar_labels = []
    s_psi_vals = []
    s_phi_vals = []
    for n in sorted(n_groups):
        for p in n_groups[n]:
            bar_labels.append(f"n={p['n']}\ng={p['g']:g}")
            s_psi_vals.append(max(p["S_psi"], 1e-4))  # log floor
            s_phi_vals.append(max(p["S_phi"], 1e-4))

    x = list(range(len(bar_labels)))
    w = 0.4
    ax_l.bar([i - w/2 for i in x], s_psi_vals, width=w,
             label=r"$S(|\psi\rangle)$ -- plain DMRG", color="#888888")
    ax_l.bar([i + w/2 for i in x], s_phi_vals, width=w,
             label=r"$S(|\phi\rangle)$ -- var-D", color="#3070f0")
    ax_l.set_yscale("log")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(bar_labels, fontsize=7, rotation=30, ha="right")
    ax_l.set_ylabel("Half-cut entanglement entropy (nats)")
    ax_l.set_title("variational-D drops |φ⟩ entropy by 4×–1430×")
    ax_l.legend(fontsize=8, loc="lower right")
    ax_l.grid(True, alpha=0.3, which="both", axis="y")

    # Right panel: entropy reduction ratio
    cmap = plt.cm.viridis
    n_values = sorted(n_groups)
    for i, n in enumerate(n_values):
        gs = [p["g"] for p in n_groups[n]]
        ratios = [p["S_psi"] / max(p["S_phi"], 1e-9) for p in n_groups[n]]
        color = cmap(i / max(1, len(n_values) - 1))
        ax_r.plot(gs, ratios, "o-", color=color, label=f"n = {n}", linewidth=1.5)
    ax_r.axvline(x=1.0, color="red", linestyle=":", alpha=0.5,
                 label="g = 1 (critical)")
    ax_r.set_yscale("log")
    ax_r.set_xlabel("Transverse field g")
    ax_r.set_ylabel(r"$S(|\psi\rangle) / S(|\phi\rangle)$  (entropy reduction)")
    ax_r.set_title("var-D entropy advantage across the TFIM phase diagram")
    ax_r.legend(fontsize=8, loc="upper right")
    ax_r.grid(True, alpha=0.3, which="both")

    fig.suptitle("variational-D CA-MPS: entropy reduction vs plain DMRG (TFIM, OBC)",
                 y=1.02, fontsize=11)
    fig.tight_layout()

    out_dir = Path(__file__).parent / "figs"
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / f"{out_basename}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"{out_basename}.pdf", bbox_inches="tight")
    print(f"wrote {out_dir / out_basename}.png and .pdf")


if __name__ == "__main__":
    here = Path(__file__).parent
    project_root = here.parent.parent.parent
    default_json = project_root / "benchmarks" / "results" / "ca_mps_var_d_vs_plain_dmrg_2026-04-29.json"

    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_json
    if not json_path.exists():
        sys.stderr.write(f"JSON not found: {json_path}\n")
        sys.exit(1)

    points = load(json_path)
    make_figure(points)
