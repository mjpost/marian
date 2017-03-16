#/bin/bash

all:
	nvcc conv_moje.cu -lcuda -lcudnn -Wno-deprecated-gpu-targets --std=c++11 -I.. -I../3rd_party -o conv_moje
