Exercise 6: Prefix Sums with MPI_Scan
========================================

What was changed from Exercise 5:
- Replaced MPI_Allreduce with MPI_Scan. The result is stored in prefix_sum.
- Unlike Allreduce (where every process gets the SAME result), MPI_Scan gives
  each process a DIFFERENT result: the cumulative sum from rank 0 through
  its own rank.
- Each process prints: local_sum, prefix_sum, and sum_before_me (prefix_sum - local_sum).
- Verification: the last rank's prefix_sum equals the global total (500,000,500,000).
- Bonus verification: each process validates prefix_sum against K*(K+1)/2 where
  K = (rank + 1) * chunk_size.

Compile and Run (on EC2):
  $ mpicc -o sum_scan sum_scan.c
  $ mpirun -np 4 ./sum_scan

Sample Output (4 processes):
  Root filled array with values 1 to 1000000
    Rank 0: local_sum = 31250125000 | prefix_sum = 31250125000 | sum_before_me = 0
    Rank 0: expected_prefix = 31250125000 | match = YES
    Rank 1: local_sum = 93750125000 | prefix_sum = 125000250000 | sum_before_me = 31250125000
    Rank 1: expected_prefix = 125000250000 | match = YES
    Rank 2: local_sum = 156250125000 | prefix_sum = 281250375000 | sum_before_me = 125000250000
    Rank 2: expected_prefix = 281250375000 | match = YES
    Rank 3: local_sum = 218750125000 | prefix_sum = 500000500000 | sum_before_me = 281250375000
    Rank 3: expected_prefix = 500000500000 | match = YES

  [Scan] Last rank prefix_sum (global total) = 500000500000
  [Scan] Expected  = 500000500000
  [Scan] Correct?  = YES
  [Scan] Time      = 0.0094 sec

Key Observations:
- MPI_Scan computes a prefix reduction. Each process gets the cumulative sum
  of all local_sums from rank 0 up to and including its own rank.
- sum_before_me acts as a global offset. This is useful for parallel file
  writing, global index assignment, cumulative distributions, and
  load-balanced work splitting — all without any extra communication.
- The bonus verification confirms each prefix_sum matches K*(K+1)/2, which
  validates that the prefix sums are globally correct.
- The last rank's prefix_sum is always equal to the global total, which
  provides the same information as MPI_Reduce on the last rank.
