/*
 * Exercise 3: Replace Send/Recv with Gather
 *
 * Changes from Exercise 2 (Scatter version):
 *  - The manual MPI_Send/MPI_Recv collection loop is replaced with
 *    a single MPI_Gather call.
 *  - Root allocates an all_sums array of size `size` to receive
 *    one long long from each process.
 *  - Root then loops through all_sums to compute the total.
 *  - Note: Gather only COLLECTS data, it does NOT reduce/sum it.
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
     * GATHER: Collect one local_sum from every process into all_sums on root.
     * all_sums is placed in rank order: all_sums[r] = local_sum of rank r.
     * Non-root processes pass NULL for recvbuf (only meaningful on root).
     *
     * Note: Gather does NOT perform any computation.
     * Root must still manually sum all_sums[] after the call.
     */
    long long *all_sums = NULL;
    if (rank == 0)
        all_sums = (long long *)malloc(size * sizeof(long long));

    MPI_Gather(&local_sum, 1, MPI_LONG_LONG,   /* each process sends 1 value */
               all_sums,  1, MPI_LONG_LONG,   /* root receives 1 per process */
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        long long total_sum = 0;
        for (int r = 0; r < size; r++)
            total_sum += all_sums[r];

        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Gather] Total sum = %lld\n", total_sum);
        printf("[Gather] Expected  = %lld\n", expected);
        printf("[Gather] Correct?  = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Gather] Time      = %.4f sec\n", elapsed);

        free(all_sums);
    }

    free(local_chunk);
    if (rank == 0) free(array);
    MPI_Finalize();
    return 0;
}
