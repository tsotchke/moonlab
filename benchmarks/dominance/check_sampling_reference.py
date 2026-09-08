"""Bounded multi-seed reproduction of issue21 with analytic detector checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--shots', type=int, default=200000)
parser.add_argument('--seeds', type=int, nargs='+', default=[1234, 7, 47, 12345, 987654321])
parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4])
args = parser.parse_args()
if not 1 <= args.shots <= 1000000 or not args.seeds or len(args.seeds) > 32:
    parser.error('shots must be 1..1000000, with 1..32 seeds')
if any(t < 1 or t > 4 for t in args.threads):
    parser.error('threads must be 1..4')
if any(s < 0 or s >= 2**64-1 for s in args.seeds):
    parser.error('seeds must leave room for the Stim seed+1')
repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))
from benchmarks.dominance.fronts import f3_batch_sampling_vs_stim as front

identity = json.loads(subprocess.check_output(
    [sys.executable, str(repo/'scripts/moonlab_source_identity.py'), '--repo-root', str(repo)],
    text=True))
n, ops, dets = front.build_surface_code_noisy(5, 8)
results = []
for threads in args.threads:
    for seed in args.seeds:
        ok, detail = front.check_detector_correctness(n, ops, dets, args.shots, seed, threads)
        results.append({'seed': seed, 'threads': threads, 'passed': ok, **detail})
nn, nops, ndets = front.build_surface_code_noisy(5, 8, p2=0, p1=0, pm=0)
silence = front.check_detector_silence(nn, nops, ndets)
passed = all(row['passed'] for row in results) and silence == (0, 0)
print(json.dumps({
    'schema': 'moonlab.sampling_reference_probe.v1', 'status': 'PASS' if passed else 'FAIL',
    'workload': 'surface_code_d5_r8_noisy', 'noise': {'p1': .001, 'p2': .001, 'pm': .001},
    'source': identity, 'library_sha256': hashlib.sha256(Path(front._LIB._name).read_bytes()).hexdigest(),
    'stim_version': front.stim.__version__, 'numpy_version': front.np.__version__,
    'shots': args.shots, 'runs': results, 'noiseless_fired': silence,
    'scope': 'same-host detector correctness; not a release certificate or dominance verdict'
}, indent=2, sort_keys=True))
sys.exit(0 if passed else 1)
