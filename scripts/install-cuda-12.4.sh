#!/usr/bin/env bash
# Install CUDA 12.4 toolkit to a user-local prefix.
# No sudo, no system paths touched. Default prefix: $HOME/.local/cuda-12.4
#
# Usage:
#   bash scripts/install-cuda-12.4.sh                  # installs to default prefix
#   PREFIX=/some/other/path bash scripts/install-cuda-12.4.sh

set -euo pipefail

VERSION="12.4.1"
DRIVER_TAG="550.54.15"
URL="https://developer.download.nvidia.com/compute/cuda/${VERSION}/local_installers/cuda_${VERSION}_${DRIVER_TAG}_linux.run"
RUNFILE="/tmp/cuda_${VERSION}_${DRIVER_TAG}_linux.run"
PREFIX="${PREFIX:-$HOME/.local/cuda-12.4}"

echo "==> Installing CUDA ${VERSION} toolkit to: ${PREFIX}"

if [[ -x "${PREFIX}/bin/nvcc" ]]; then
    echo "==> ${PREFIX}/bin/nvcc already exists; skipping install."
    "${PREFIX}/bin/nvcc" --version
    exit 0
fi

if [[ ! -f "${RUNFILE}" ]]; then
    echo "==> Downloading runfile (~4 GB) to ${RUNFILE}"
    wget -O "${RUNFILE}" "${URL}"
fi

mkdir -p "${PREFIX}"

# --silent --toolkit installs only the toolkit (no driver, no GL stubs).
# --override skips the bundled-driver compatibility check.
# --installpath sets where the toolkit lives.
# --no-man-page / --no-opengl-libs trim more bytes off the install.
sh "${RUNFILE}" \
    --silent \
    --toolkit \
    --no-opengl-libs \
    --no-man-page \
    --override \
    --toolkitpath="${PREFIX}"

echo
echo "==> Done. nvcc reports:"
"${PREFIX}/bin/nvcc" --version
echo
echo "==> CUDA 12.4 also requires a host compiler no newer than gcc 13."
echo "    On Arch:  sudo pacman -S gcc13"
echo
echo "==> Set this in your shell before running 'uv sync':"
echo "    export CUDA_HOME=${PREFIX}"
echo "    export PATH=${PREFIX}/bin:\$PATH"
echo "    export CC=/usr/bin/gcc-13"
echo "    export CXX=/usr/bin/g++-13"
echo "    export NVCC_PREPEND_FLAGS=\"-ccbin /usr/bin/g++-13\""
