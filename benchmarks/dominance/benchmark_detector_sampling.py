"""Fresh-process detector latency/RSS probe with explicit output contracts.

Run each engine in a separate invocation. This does not certify correctness
or a performance win; use the independent sampling gates first. RSS includes
Python, NumPy, Stim and the loaded Moonlab library in every invocation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--library', type=Path, required=True)
parser.add_argument('--engine', choices=['moonlab', 'stim-unpacked', 'stim-packed'], required=True)
parser.add_argument('--shots', type=int, default=200000)
parser.add_argument('--distance', type=int, default=5)
parser.add_argument('--rounds', type=int, default=8)
parser.add_argument('--threads', type=int, default=1)
parser.add_argument('--repetitions', type=int, default=9)
args = parser.parse_args()
if not (1 <= args.shots <= 1000000 and 2 <= args.distance <= 9 and 1 <= args.rounds <= 16):
    parser.error('workload exceeds this bounded probe')
if not 1 <= args.threads <= 4 or not 3 <= args.repetitions <= 31:
    parser.error('threads must be 1..4 and repetitions 3..31')
if args.engine != 'moonlab' and args.threads != 1:
    parser.error('Stim sampling uses one thread')
library = args.library.resolve(strict=True)
os.environ['MOONLAB_LIB_DIR'] = str(library.parent)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import stim
from benchmarks.dominance.fronts import f3_batch_sampling_vs_stim as front
if Path(front._LIB._name).resolve() != library:
    raise RuntimeError('the loader selected a different library')

def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)

n, ops, detectors = front.build_surface_code_noisy(args.distance, args.rounds)
if len(detectors) * args.shots > 256000000:
    parser.error('public output exceeds 256 MB')
start = time.perf_counter_ns()
if args.engine == 'moonlab':
    op_array = front._pf_op_array(ops)
    offsets, indices = front._det_csr(detectors)
    import ctypes
    def sample():
        out = np.empty((len(detectors), args.shots), dtype=np.uint8)
        count = front._LIB.pauli_frame_batch_sample_detectors(
            n, op_array, len(ops),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            len(detectors), args.shots, 1234, args.threads,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        if count != len(detectors):
            raise RuntimeError(f'sampler returned {count}, expected {len(detectors)}')
        return out
else:
    sampler = front._to_stim_with_detectors(ops, detectors).compile_detector_sampler(seed=1235)
    def sample():
        return sampler.sample(args.shots, bit_packed=args.engine == 'stim-packed')
prepare_ms = (time.perf_counter_ns() - start) / 1e6
before_rss = peak_rss_bytes()
durations = []
output_bytes = None
for repetition in range(args.repetitions + 1):
    start = time.perf_counter_ns()
    out = sample()
    duration = (time.perf_counter_ns() - start) / 1e6
    output_bytes = out.nbytes
    del out  # retain exactly one output at a time in every engine
    durations.append(duration)
peak = peak_rss_bytes()
warm = durations[1:]
print(json.dumps({
    'engine': args.engine, 'shots': args.shots, 'distance': args.distance,
    'rounds': args.rounds, 'detectors': len(detectors), 'requested_threads': args.threads,
    'stim_version': stim.__version__, 'numpy_version': np.__version__,
    'platform': platform.platform(), 'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
    'output_layout': 'detector-major bytes' if args.engine == 'moonlab' else
                     ('shot-major packed bits' if args.engine == 'stim-packed' else 'shot-major bytes'),
    'output_bytes': output_bytes, 'prepare_ms': prepare_ms, 'cold_sample_ms': durations[0],
    'warm_median_ms': statistics.median(warm),
    'warm_p95_ms': float(np.percentile(warm, 95)), 'warm_samples_ms': warm,
    'process_peak_rss_bytes': peak, 'pre_sampling_peak_rss_bytes': before_rss,
    'peak_rss_increase_bytes': max(0, peak-before_rss),
    'scope': 'local probe, co-resident host; Moonlab reference setup included per sample; no dominance certification'
}, sort_keys=True))
