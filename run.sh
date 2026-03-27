#!/bin/bash
# AArch64 cross-build only
set -e

export TOOLCHAIN_PATH="/opt/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu"
export PATH="${TOOLCHAIN_PATH}/bin:${PATH}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD="${ROOT}/build"

mkdir -p "${BUILD}"

echo "Configuring (AArch64 cross)..."
CMAKE_ARGS=(
  -S "${ROOT}"
  -B "${BUILD}"
  -DCMAKE_TOOLCHAIN_FILE="${ROOT}/toolchain-aarch64.cmake"
  -DCMAKE_BUILD_TYPE=Release
)
if [ -n "${SYSROOT:-}" ]; then
  CMAKE_ARGS+=( -DCMAKE_SYSROOT="${SYSROOT}" )
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD}" -j"$(nproc 2>/dev/null || echo 4)"

echo "Build completed successfully."
