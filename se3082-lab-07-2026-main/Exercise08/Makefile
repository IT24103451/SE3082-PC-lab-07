# Makefile for SE3082 Lab 07 - MPI Collective Communication
# Usage:
#   make         - compile all 6 programs
#   make run     - run all 6 programs with 4 processes
#   make clean   - remove all compiled binaries

CC     = mpicc
CFLAGS = -O2 -Wall

# All executables
PROGS = sum_scatter sum_gather sum_reduce sum_allreduce sum_scan

.PHONY: all run clean

all: $(PROGS)

sum_scatter: ../Exercise02/sum_scatter.c
	$(CC) $(CFLAGS) -o sum_scatter ../Exercise02/sum_scatter.c

sum_gather: ../Exercise03/sum_gather.c
	$(CC) $(CFLAGS) -o sum_gather ../Exercise03/sum_gather.c

sum_reduce: ../Exercise04/sum_reduce.c
	$(CC) $(CFLAGS) -o sum_reduce ../Exercise04/sum_reduce.c

sum_allreduce: ../Exercise05/sum_allreduce.c
	$(CC) $(CFLAGS) -o sum_allreduce ../Exercise05/sum_allreduce.c

sum_scan: ../Exercise06/sum_scan.c
	$(CC) $(CFLAGS) -o sum_scan ../Exercise06/sum_scan.c

run: all
	@echo "=============================="
	@echo " Running sum_scatter  (np=4)"
	@echo "=============================="
	mpirun -np 4 ./sum_scatter
	@echo ""
	@echo "=============================="
	@echo " Running sum_gather   (np=4)"
	@echo "=============================="
	mpirun -np 4 ./sum_gather
	@echo ""
	@echo "=============================="
	@echo " Running sum_reduce   (np=4)"
	@echo "=============================="
	mpirun -np 4 ./sum_reduce
	@echo ""
	@echo "=============================="
	@echo " Running sum_allreduce (np=4)"
	@echo "=============================="
	mpirun -np 4 ./sum_allreduce
	@echo ""
	@echo "=============================="
	@echo " Running sum_scan     (np=4)"
	@echo "=============================="
	mpirun -np 4 ./sum_scan

clean:
	rm -f $(PROGS)
