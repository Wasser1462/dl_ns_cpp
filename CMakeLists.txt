cmake_minimum_required(VERSION 3.13)
project(dl_ns_cpp CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src
)

add_executable(dtln
  src/main.cc
  src/RWwav.cc
  src/dtln_process.cc
  src/sfft.cc
)

target_compile_definitions(dtln PRIVATE
  TF_MAJOR_VERSION=2
  TF_MINOR_VERSION=6
  TF_PATCH_VERSION=0
  TF_VERSION_SUFFIX=""
)

if(FULLY_STATIC)
  set(BUILD_SHARED_LIBS OFF)
  set(CMAKE_SKIP_RPATH TRUE)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  target_link_options(dtln PRIVATE -static -s)
else()
  target_link_options(dtln PRIVATE -static-libstdc++ -static-libgcc -s)
endif()

set(DTLN_LIB_DIR ${CMAKE_SOURCE_DIR}/libs)
set(TFLITE_LIB ${DTLN_LIB_DIR}/libtensorflow-lite.a)
set(FFTW3F_LIB ${DTLN_LIB_DIR}/libfftw3f.a)

target_link_libraries(dtln
  PRIVATE
    ${TFLITE_LIB}
    ${FFTW3F_LIB}
    pthread
    dl
    atomic
    m
)
