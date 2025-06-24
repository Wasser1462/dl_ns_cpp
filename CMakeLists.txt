cmake_minimum_required(VERSION 3.16)
project(DTLN_Denoise VERSION 1.0 LANGUAGES C CXX)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    execute_process(COMMAND ${CCACHE_PROGRAM} -M 2G)
endif()

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_definitions(
    -DTF_MAJOR_VERSION=2
    -DTF_MINOR_VERSION=6
    -DTF_PATCH_VERSION=0
    -DTF_VERSION_SUFFIX=""
)

add_subdirectory(
    libs/tensorflow/tensorflow/lite 
    ${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite  
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)
if(NOT SNDFILE_FOUND)
    set(SNDFILE_INCLUDE_DIRS "/usr/include")
    set(SNDFILE_LIBRARIES "/usr/lib64/libsndfile.so")
    message(WARNING "Using hardcoded sndfile paths: ${SNDFILE_LIBRARIES}")
endif()

add_executable(dtln 
    src/dtln.cc
    src/sfft.cc
    src/RWwav.cc
)

target_precompile_headers(dtln PRIVATE 
    src/stdafx.h
)

target_include_directories(dtln PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}        
    ${CMAKE_CURRENT_SOURCE_DIR}/libs    
    ${SNDFILE_INCLUDE_DIRS}
)

target_link_libraries(dtln
    tensorflow-lite
    ${SNDFILE_LIBRARIES}
    pthread
    dl
)

target_link_options(dtln PRIVATE -fsanitize=address)

install(TARGETS dtln RUNTIME DESTINATION bin)