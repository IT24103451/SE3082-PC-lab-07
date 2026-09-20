/*
 * Exercise 5: Replace Reduce with Allreduce
 *
 * Changes from Exercise 4 (Reduce version):
 *  - MPI_Reduce is replaced with MPI_Allreduce (no root parameter).
 *  - Every process receives the correct total_sum (not just root).
 *  - Every process prints its local_sum, total_sum, and percentage contribution.
 *  - Root still prints the final verification.
 *
 * Best practice: Prefer MPI_Allreduce over MPI_Reduce + MPI_Bcast when
 * all processes need the result. It uses a more efficient butterfly/
 * recursive-doubling algorithm internally.
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
     * ALLREDUCE: Like MPI_Reduce but the result is delivered to ALL processes.
     * No root parameter. Every rank's total_sum will hold the global sum.
     *
     * This eliminates the need for a follow-up MPI_Bcast after MPI_Reduce.
     */
    long long total_sum = 0;
    MPI_Allreduce(&local_sum,   /* sendbuf: each process contributes  */
                  &total_sum,   /* recvbuf: result goes to ALL ranks  */
                  1,            /* count                              */
                  MPI_LONG_LONG,/* datatype                           */
                  MPI_SUM,      /* op                                 */
                  MPI_COMM_WORLD);

    /*
     * Every process now has the correct total_sum.
     * Each process prints its contribution and percentage.
     */
    double percentage = (double)local_sum / (double)total_sum * 100.0;
    printf("  Rank %d: local_sum = %lld  |  total_sum = %lld  |  contribution = %.4f%%\n",
           rank, local_sum, total_sum, percentage);

    /* Root prints the final verification */
    if (rank == 0) {
        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Allreduce] Total sum = %lld\n", total_sum);
        printf("[Allreduce] Expected  = %lld\n", expected);
        printf("[Allreduce] Correct?  = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Allreduce] Time      = %.4f sec\n", elapsed);
    }

    free(local_chunk);
    if (rank == 0) free(array);
    MPI_Finalize();
    return 0;
}
