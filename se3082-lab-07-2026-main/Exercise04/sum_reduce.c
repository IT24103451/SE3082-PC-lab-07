/*
 * Exercise 4: Replace Gather with Reduce
 *
 * Changes from Exercise 3 (Gather version):
 *  - The all_sums array and the manual summation loop on root are removed.
 *  - MPI_Gather is replaced with MPI_Reduce using MPI_SUM.
 *  - MPI internally computes the sum using a tree-based algorithm O(log P).
 *  - After MPI_Reduce, total_sum on root already contains the correct answer.
 *
 * Important: total_sum is only valid on root (rank 0) after MPI_Reduce.
 * Non-root processes have an undefined/garbage value in their recvbuf.
 * If all processes need the result, use MPI_Allreduce (see Exercise 5).
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

    printf("  Rank %d: local_sum = %lld\n", rank, local_sum);

    /*
     * REDUCE: Each process contributes local_sum.
     * MPI applies MPI_SUM across all processes using a tree algorithm.
     * The final sum is stored in total_sum on root (rank 0) only.
     *
     * recvbuf (total_sum) is only valid on root after this call.
     * No all_sums array needed; no manual loop needed on root.
     */
    long long total_sum = 0;
    MPI_Reduce(&local_sum,  /* sendbuf: value each process contributes */
               &total_sum,  /* recvbuf: result stored here (root only) */
               1,           /* count: one element to reduce            */
               MPI_LONG_LONG, /* datatype                              */
               MPI_SUM,     /* op: sum all local_sums                  */
               0,           /* root: rank 0 receives the result        */
               MPI_COMM_WORLD);

    if (rank == 0) {
        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Reduce] Total sum = %lld\n", total_sum);
        printf("[Reduce] Expected  = %lld\n", expected);
        printf("[Reduce] Correct?  = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Reduce] Time      = %.4f sec\n", elapsed);
    }

    free(local_chunk);
    if (rank == 0) free(array);
    MPI_Finalize();
    return 0;
}
