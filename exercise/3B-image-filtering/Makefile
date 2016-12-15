#!/bin/bash

include ../Makefile.inc

SCR = $(wildcard *.cu)
EXE = $(SCR:.cu=.x)

default: $(EXE)

%.x : %.cu
	$(CC) $(CFLAGS) $< -o $@

clean:
	$(RM) $(EXE) result.png
