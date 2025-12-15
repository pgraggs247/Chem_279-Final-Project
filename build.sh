#!/bin/sh

mkdir -p build
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j4

# make the run script executable
chmod +x ../run.sh