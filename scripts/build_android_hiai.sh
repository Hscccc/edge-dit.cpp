#!/usr/bin/env bash
set -euo pipefail
: "${ANDROID_NDK_HOME:?Set ANDROID_NDK_HOME to Android NDK r27 or compatible}"
task_root=$(cd "$(dirname "$0")/.." && pwd)
cmake -S "$task_root" -B "$task_root/build-android-hiai" \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF -DGGML_BACKEND_DL=OFF -DGGML_CPU_KLEIDIAI=OFF \
  -DED_GGML_HIAI=ON -DED_BUILD_MOBILE_TESTS=ON
cmake --build "$task_root/build-android-hiai" --target ed-cli ed-test-hiai ed-hiai-probe -j "${BUILD_JOBS:-6}"
