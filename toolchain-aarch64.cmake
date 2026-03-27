# toolchain-aarch64.cmake
# Arm GNU aarch64-none-linux-gnu（与 sherpa-onnx/toolchains/arm-gnu-aarch64.toolchain.cmake 同布局）
set(CMAKE_SYSTEM_NAME      Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(TOOLCHAIN_DIR "/opt/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu")
set(CMAKE_C_COMPILER  ${TOOLCHAIN_DIR}/bin/aarch64-none-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-none-linux-gnu-g++)
set(CMAKE_SYSROOT ${TOOLCHAIN_DIR}/aarch64-none-linux-gnu/libc)

# 交叉编译时限制在 sysroot 内查找库/头文件（给 sherpa-onnx / 其它 CMake 工程用）
set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
