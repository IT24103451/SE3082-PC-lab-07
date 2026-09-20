/*
 * Exercise 6: Prefix Sums with MPI_Scan
 *
 * Changes from Exercise 5 (Allreduce version):
 *  - MPI_Allreduce is replaced with MPI_Scan.
 *  - Unlike Allreduce (same result for all), each process gets a DIFFERENT
 *    result: the cumulative sum from rank 0 through its own rank.
 *
 * After MPI_Scan with MPI_SUM:
 *   Rank 0: prefix_sum = local_sum_0
 *   Rank 1: prefix_sum = local_sum_0 + local_sum_1
 *   Rank 2: prefix_sum = local_sum_0 + local_sum_1 + local_sum_2
 *   Last rank: prefix_sum = total sum of entire array (500,000,500,000)
 *
 * Practical use: sum_before_me acts as a global offset, enabling globally
 * correct running totals, index assignment, and cumulative distributions
 * without extra communication passes.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = N / size;

    /* Only root allocates the full array */
    int *array = NULL;
    if (rank == 0) {
        array = (int *)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++)
            array[i] = i + 1;
        printf("Root filled array with values 1 to %d\n", N);
    }

    /* Each process allocates only its local chunk */
    int *local_chunk = (int *)malloc(chunk_size * sizeof(int));

    double start = MPI_Wtime();

    /* Distribute chunks via Scatter */
    MPI_Scatter(array, chunk_size, MPI_INT,
                local_chunk, chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    /* Each process computes its partial sum */
    long long local_sum = 0;
    for (int i = 0; i < chunk_size; i++)
        local_sum += local_chunk[i];

    /*
     * SCAN (Prefix Reduction):
     * Each process receives the cumulative sum of all ranks from 0 up to
     * and including its own rank. No root parameter needed.
     *
     * Result differs per process (unlike Allreduce where all get the same).
     */
    long long prefix_sum = 0;
    MPI_Scan(&local_sum,   /* sendbuf: this rank's contribution      */
             &prefix_sum,  /* recvbuf: cumulative sum up to this rank */
             1,            /* count                                   */
             MPI_LONG_LONG,/* datatype                                */
             MPI_SUM,      /* op                                      */
             MPI_COMM_WORLD);

    /*
     * sum_before_me = total of all chunks BEFORE this rank.
     * Useful as a global offset for index assignment.
     */
    long long sum_before_me = prefix_sum - local_sum;

    printf("  Rank %d: local_sum = %lld | prefix_sum = %lld | sum_before_me = %lld\n",
           rank, local_sum, prefix_sum, sum_before_me);

    /*
     * Verification bonus: For array 1..N, sum of elements 1 through K is K*(K+1)/2.
     * Each process can verify its prefix_sum using K = (rank + 1) * chunk_size.
     */
    long long K = (long long)(rank + 1) * chunk_size;
    long long expected_prefix = K * (K + 1) / 2;
    printf("  Rank %d: expected_prefix = %lld | match = %s\n",
           rank, expected_prefix, prefix_sum == expected_prefix ? "YES" : "NO");

    /* Verification: last rank's prefix_sum must equal the global total */
    if (rank == size - 1) {
        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Scan] Last rank prefix_sum (global total) = %lld\n", prefix_sum);
        printf("[Scan] Expected  = %lld\n", expected);
        printf("[Scan] Correct?  = %s\n", prefix_sum == expected ? "YES" : "NO");
        printf("[Scan] Time      = %.4f sec\n", elapsed);
    }

    free(local_chunk);
    if (rank == 0) free(array);
    MPI_Finalize();
    return 0;
}
