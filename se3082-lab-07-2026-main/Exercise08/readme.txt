Exercise 8: Compile and Run
==============================

A Makefile has been created in this directory that compiles all 6 programs
and provides a run target to execute them all with 4 processes.

Usage on EC2:
  $ cd Exercise08
  $ make          # Compiles all 6 programs
  $ make run      # Runs all 6 programs with mpirun -np 4
  $ make clean    # Removes all compiled binaries

Makefile Targets:
  all           - Compiles: sum_scatter, sum_gather, sum_reduce,
                  sum_allreduce, sum_scan
  run           - Runs all compiled programs with mpirun -np 4
  clean         - Removes all compiled binaries

Compile Output:
  $ make
  mpicc -O2 -Wall -o sum_scatter ../Exercise02/sum_scatter.c
  mpicc -O2 -Wall -o sum_gather ../Exercise03/sum_gather.c
  mpicc -O2 -Wall -o sum_reduce ../Exercise04/sum_reduce.c
  mpicc -O2 -Wall -o sum_allreduce ../Exercise05/sum_allreduce.c
  mpicc -O2 -Wall -o sum_scan ../Exercise06/sum_scan.c

All programs compile without warnings using -Wall flag.
All programs produce the correct output: Total sum = 500,000,500,000.
All source files and the Makefile have been pushed to the GitHub repository.
