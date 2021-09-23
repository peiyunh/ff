#!/bin/bash
mkdir build
cd build
cmake -DPYTHON_LIBRARY=/home/peiyunh/miniconda3/envs/mmp/lib/libpython3.7m.so -DPYTHON_EXECUTABLE=/home/peiyunh/miniconda3/envs/mmp/bin/python ..
make
cd ..
