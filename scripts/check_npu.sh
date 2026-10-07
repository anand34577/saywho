#!/usr/bin/env bash
# Run on the board (host, not in the container):  sudo bash scripts/check_npu.sh
# librknnrt needs Rockchip's vendor "rknpu" kernel driver (Armbian "vendor" kernel, 5.10 / 6.1).
# The mainline "rocket" NPU driver (Armbian current/edge kernels) is NOT compatible with librknnrt.

echo "SoC:      $(tr -d '\0' < /proc/device-tree/compatible 2>/dev/null | grep -oE 'rk35[0-9]{2}s?' | head -1)"
echo "Kernel:   $(uname -r)"
echo "Arch:     $(uname -m)"

if [ -r /sys/kernel/debug/rknpu/version ]; then
  echo "rknpu:    $(cat /sys/kernel/debug/rknpu/version)"
  echo "NPU load: $(cat /sys/kernel/debug/rknpu/load 2>/dev/null)"
elif dmesg 2>/dev/null | grep -qi 'rknpu'; then
  echo "rknpu:    $(dmesg | grep -i 'rknpu' | grep -i -m1 'version\|initialized')"
else
  echo "rknpu:    NOT FOUND -> install/boot the Armbian *vendor* kernel (e.g. armbian-config > System > Kernel)"
fi

if lsmod 2>/dev/null | grep -q '^rocket'; then
  echo "WARNING:  mainline 'rocket' NPU driver is loaded; librknnrt will not work with it"
fi

echo "DRM nodes: $(ls /dev/dri 2>/dev/null | tr '\n' ' ')"
command -v docker >/dev/null && echo "Docker:   $(docker --version)" || echo "Docker:   not installed"
