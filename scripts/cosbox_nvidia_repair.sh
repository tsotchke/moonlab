#!/usr/bin/env bash
# cosbox_nvidia_repair.sh -- bring cosbox (RTX 3090, Ubuntu) back online
#                            after the in-flight driver upgrade that left
#                            kernel module + userspace at the same version
#                            string but in incompatible runtime states.
#
# Symptom: `nvidia-smi` returns
#     Failed to initialize NVML: Driver/library version mismatch
# even though `modinfo nvidia | grep version` and
# `ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.595.*` agree on the
# driver string.  This happens when the apt upgrade installed new
# files on disk while the OLD nvidia kernel module is still loaded
# in the running kernel.  Userspace + kernel can't talk because the
# kernel side is the previous version.
#
# Fix is a clean reboot.  Reloading the module without rebooting
# would require stopping every GPU consumer (X11, lightdm, any
# CUDA process), which is more invasive than a reboot.
#
# Usage:
#   ssh tyr@cosbox  'bash -s' < scripts/cosbox_nvidia_repair.sh
#   OR
#   scp scripts/cosbox_nvidia_repair.sh tyr@cosbox:/tmp/
#   ssh tyr@cosbox 'sudo bash /tmp/cosbox_nvidia_repair.sh'

set -euo pipefail

echo "=== Pre-repair state ==="
uname -a
echo
echo "--- nvidia-smi (expected: mismatch error) ---"
nvidia-smi 2>&1 | head -3 || true
echo
echo "--- /proc/driver/nvidia/version (kernel module currently loaded) ---"
cat /proc/driver/nvidia/version 2>/dev/null || echo "(kernel module not loaded)"
echo
echo "--- modinfo nvidia (kernel module on disk after package upgrade) ---"
modinfo nvidia 2>/dev/null | grep -E "version|filename" | head -3 || true
echo
echo "--- libnvidia-ml userspace ---"
ls /usr/lib/x86_64-linux-gnu/libnvidia-ml* 2>/dev/null | head -3 || true
echo

# Sanity: confirm the kernel module on disk is what we expect.
KERN_VER_ONDISK=$(modinfo nvidia 2>/dev/null | awk '$1 == "version:" {print $2; exit}' || true)
USERLAND_LIB=$(ls -1 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.* 2>/dev/null \
              | grep -E "libnvidia-ml.so\\.[0-9]+\\.[0-9]+\\.[0-9]+$" \
              | head -1 || true)
USERLAND_VER=${USERLAND_LIB##*libnvidia-ml.so.}

if [[ -n "$KERN_VER_ONDISK" && -n "$USERLAND_VER" \
      && "$KERN_VER_ONDISK" != "$USERLAND_VER" ]]; then
    echo "WARN: kernel module on disk ($KERN_VER_ONDISK) and userspace"
    echo "WARN: libnvidia-ml ($USERLAND_VER) DON'T match.  A reboot alone"
    echo "WARN: won't be enough -- the apt upgrade was probably partial."
    echo "WARN: Recommend running:"
    echo "WARN:    sudo apt-get install --reinstall -y nvidia-driver-595-open"
    echo "WARN:    sudo reboot"
    echo
fi

echo "=== Reboot plan ==="
echo "Will reboot in 10 seconds.  Ctrl-C now to abort."
sleep 10

# Use systemd-friendly reboot rather than `reboot -f` so currently-running
# services get a graceful shutdown.  The bash exits before this returns.
exec sudo systemctl reboot

# Unreachable but kept for clarity:
echo "(reboot dispatched)"
