#!/bin/sh
# Entry point of the rknn image. Rockchip's NPU runtime is not part of the image (proprietary RKNN SDK License):
# with RKNN_ACCEPT_LICENSE=yes it is downloaded from Rockchip on first start into /opt/rknn (mount a volume there to
# keep it). Without it, or if the download fails, the server still starts and runs on the CPU.
LICENSE_URL=https://github.com/airockchip/rknn-toolkit2/blob/master/LICENSE

case "$(echo "${RKNN_ACCEPT_LICENSE:-no}" | tr 'A-Z' 'a-z')" in
  yes|true|1)
    python /app/scripts/fetch_rknn_runtime.py --accept-license --target /opt/rknn \
      || echo "WARNING: could not download the Rockchip NPU runtime; starting on the CPU" >&2 ;;
  *)
    if [ ! -f /opt/rknn/lib/librknnrt.so ]; then
      echo "NOTICE: the Rockchip NPU runtime is not installed. It is licensed by Rockchip under the RKNN SDK License" >&2
      echo "        ($LICENSE_URL). Set RKNN_ACCEPT_LICENSE=yes to accept it and download it on start." >&2
      echo "        Until then the server runs on the CPU." >&2
    fi ;;
esac

exec "$@"
