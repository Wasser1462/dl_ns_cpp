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
  src/wav_io.cc
  src/dtln_process.cc
  src/sfft.cc
)

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
