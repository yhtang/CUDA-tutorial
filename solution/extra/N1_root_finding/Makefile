#!/bin/bash

include ../Makefile.inc

SCR = $(wildcard *.cu)
EXE = $(SCR:.cu=.x)

default: $(EXE)

%.x : %.cu
	$(CC) $(CFLAGS) $< -o $@ $(DPFLAGS)

clean:
	$(RM) $(EXE)
