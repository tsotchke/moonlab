# Quarantined findings (excluded from the replay PASS gate)

Files here are genuine findings pending a fix in a subsystem this lane
does not own. `scripts/run_fuzz.sh replay` deliberately skips this
subdirectory (it enumerates only top-level corpus files), so the required
replay gate stays green until the owning lane lands a fix; drop the
reproducer up one level into the seed corpus once it is fixed so the gate
then guards against regressions.

## oom_qubit_amplification.bin  ->  src/control/control_plane.c (+ src/applications/moonlab_qgtl_backend.c)

Wire bytes: `CIRCUIT 18\nNUM_QUBITS 30\nH 0\n`

A single, well-formed CIRCUIT frame under the 4 MB body cap declares a
30-qubit circuit. `handle_one_request` deserializes it and calls
`moonlab_qgtl_execute`, which allocates a `2^num_qubits` state vector --
2^30 * 16 bytes = 16 GB here, and up to 64 GB at the deserializer's
`num_qubits == 32` ceiling. There is no pre-execution qubit/size cap in
the public build, so one small request drives an unbounded server-side
allocation (memory-exhaustion DoS / amplification). Surfaced by the
libFuzzer soak on the control-plane target (RSS OOM after ~40k execs).

This is a resource-amplification bug, not a memory-safety violation --
no OOB access, no UAF. See ../../../FINDINGS.md for the hand-off.
