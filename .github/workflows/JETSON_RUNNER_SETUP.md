# Archived Moonlab Documentation: Jetson CI runner — one-time setup

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Jetson CI runner — one-time setup

This document covers bringing up the `ci-jetson.yml` workflow on a
self-hosted Jetson Xavier running Anduril's jetpack-nixos.  It is
intentionally terse; the on-host portion is roughly five commands
once you have a runner registration token.

## Why self-hosted?

GitHub-hosted runners do not provide an aarch64 + on-chip CUDA
combination.  The closest hosted options are:
- `ubuntu-22.04-arm` (aarch64, no GPU)
- `ubuntu-22.04` + `nvidia/cuda` container (x86_64 + Ampere/Hopper)

The v1.1 GPU arc validates a Volta cc 7.2 GPU on aarch64 with the
L4T-pinned CUDA 11.4 toolchain.  That combination only exists on
real Jetson hardware, so the runner must live on-host.

## Prerequisites

- A working jetpack-nixos install on the Jetson.
- `nix` available as the user the runner will run as (test:
  `nix --version` should print 2.18+).
- The flake at the repo root resolves (`nix flake show` succeeds in
  the working tree).
- The runner user can read `/dev/nvidia*`.  On jetpack-nixos this
  works out of the box; on stock NVIDIA L4T you may need to add
  the user to a `video` group.

## On-host install

1. Register the runner.  From the Jetson, as the runner user, with
   the runner registration token from `Settings -> Actions ->
   Runners -> New self-hosted runner` in the public moonlab repo:

[archived fence delimiter:    ```sh]
   mkdir -p ~/actions-runner && cd ~/actions-runner
   curl -O -L https://github.com/actions/runner/releases/download/v2.319.1/actions-runner-linux-arm64-2.319.1.tar.gz
   tar xzf actions-runner-linux-arm64-2.319.1.tar.gz
   ./config.sh \
       --url https://github.com/tsotchke/moonlab \
       --token <PASTE_TOKEN_HERE> \
       --labels self-hosted,linux,ARM64,jetpack,cuda \
       --work _work \
       --name jetson-xavier-01
[archived fence delimiter:    ```]

2. Install as a systemd service so it survives reboots:

[archived fence delimiter:    ```sh]
   sudo ./svc.sh install <runner_user>
   sudo ./svc.sh start
   sudo ./svc.sh status
[archived fence delimiter:    ```]

3. Move the working directory off the small eMMC.  Xavier's onboard
   storage is typically ~32 GB; ctest artefacts + the flake.nix
   /nix/store easily exceed that.  Edit the systemd unit:

[archived fence delimiter:    ```sh]
   sudo systemctl edit actions.runner.tsotchke-moonlab.jetson-xavier-01
   # Add:
   [Service]
   WorkingDirectory=/storage/actions-runner
   Environment=RUNNER_WORK_DIR=/storage/actions-runner/_work
[archived fence delimiter:    ```]

   ...and `sudo systemctl daemon-reload && sudo systemctl restart` it.

4. Pre-warm the flake.nix dev shell so the first CI run doesn't
   spend 30 min pulling cuda-merged from nixpkgs:

[archived fence delimiter:    ```sh]
   cd /storage/actions-runner/_work/moonlab/moonlab
   nix --extra-experimental-features 'nix-command flakes' develop \
       --command bash -lc 'echo ready'
[archived fence delimiter:    ```]

## What the workflow runs

See `ci-jetson.yml`.  Per push, the workflow:

1. Configures CMake with `QSIM_ENABLE_CUDA=ON` and
   `CMAKE_CUDA_ARCHITECTURES=72` (Volta).
2. Builds libquantumsim + tests + examples.
3. Runs the three v1.1 GPU parity examples
   (`state_gpu_bell`, `state_gpu_circuit`, `state_gpu_multiq`).
4. Runs `ctest --output-on-failure -j 4`, excluding heavy tests.
5. Runs `mpirun -n 2 large_state_ghz_gpu 8` to exercise the
   v1.1 follow-up #6 sharded-MPI+CUDA path with two ranks on a
   single Volta tile (cooperative kernel launch).

Total wall time: ~6 min steady-state with a warm ccache, ~25 min
on a cold flake-eval.

## Adding more nodes

When old-donkey (RTX 3050, x86_64, Ubuntu CUDA 13.1) or cosbox
(RTX 3090) come online as runners, copy the `ci-jetson.yml` to
`ci-ampere.yml` / `ci-ada.yml` and:

- Change `runs-on:` labels to `[self-hosted, linux, X64, cuda]`
  (plus a host-specific label like `rtx-3050` or `rtx-3090`).
- Bump `CMAKE_CUDA_ARCHITECTURES` to `86` (Ampere) or `89` (Ada
  Lovelace) accordingly.
- Drop the `nix develop` wrapper since the x86_64 hosts use a
  plain apt-installed CUDA toolkit.

The `cuda` label is the conjunction of all CUDA-capable runners,
so any future workflow that says `runs-on: [self-hosted, cuda]`
will pick the first available GPU host regardless of architecture.

## Disabling the workflow

While the no-public-push policy is in effect (see
`feedback_no_public_push` in the project memory), pushing this YAML
to GitHub will trigger immediate CI runs.  Options:

1. **Don't push yet.**  The yaml will sit in the local working tree
   until the public-push policy is lifted; the runner setup steps
   above can still be completed on the Jetson now so everything is
   ready to fire once the policy is lifted.
2. **Push with the workflow temporarily gated.**  Add a top-level
   `if: github.event.head_commit.message == 'enable-ci-jetson'`
   to the job, push the workflow, then push a trigger commit when
   ready to start CI.

Option 1 is the default and matches the rest of the project's
no-push posture.
```
