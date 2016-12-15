#!/bin/bash

CC	= nvcc
CFLAGS	= -O3 -arch=sm_35 -Xcompiler -fopenmp
DPFLAGS = -rdc=true -lcudadevrt
SHELL	= /bin/bash
RM      = rm
