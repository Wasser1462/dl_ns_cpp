#!/bin/bash


TF_LIB_PATH="/data/zengyongwang/dtln_c/libs/tensorflow/tensorflow/lite"

g++ -v debug.cc \
  -I/data/zengyongwang/dtln_c/libs/tensorflow \
  -I/root/flatbuffers/include \
  -I/usr/include \
  -L${TF_LIB_PATH} \        
  -L/usr/lib64 \
  -ltensorflow-lite \                 
  -lsndfile \
  -o debug