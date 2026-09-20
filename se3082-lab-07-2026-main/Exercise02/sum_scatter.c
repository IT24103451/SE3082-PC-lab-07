/*
 * Exercise 2: Replace Broadcast with Scatter
 *
 * Changes from Exercise 1 (Bcast version):
 *  - Only root allocates the full N-element array.
 *  - All processes allocate a local_chunk of size (N / size).
 *  - MPI_Bcast is replaced with MPI_Scatter, which divides the array
 *    on root and sends ONLY the relevant chunk to each process.
 *  - The local sum loop now iterates over local_chunk[0..chunk_size).
 *  - The Send/Recv collection is kept unchanged (as per instructions).
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

    /*
     * MEMORY OPTIMIZATION:
     * Only root allocates the full array.
     * All other processes allocate NULL - they don't need the full array.
     */
    int *array = NULL;
    if (rank == 0) {
        array = (int *)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++)
            array[i] = i + 1;
        printf("Root filled array with values 1 to %d\n", N);
    }

    /*
     * Every process allocates only its local chunk buffer.
     */
    int *local_chunk = (int *)malloc(chunk_size * sizeof(int));

    double start = MPI_Wtime();

    /*
     * SCATTER: Root divides the full array into equal chunks and sends
     * one chunk to each process (including itself).
     * sendcount = chunk_size (elements sent TO EACH process, not total!)
     */
    MPI_Scatter(array,       /* sendbuf: full array on root        */
                chunk_size,  /* sendcount: elements per process    */
                MPI_INT,     /* sendtype                           */
                local_chunk, /* recvbuf: local buffer              */
                chunk_size,  /* recvcount: elements this rank gets */
                MPI_INT,     /* recvtype                           */
                0,           /* root                               */
                MPI_COMM_WORLD);

    long long local_sum = 0;
    for (int i = 0; i < chunk_size; i++)
        local_sum += local_chunk[i];

    printf("  Rank %d: summed local_chunk[0..%d) => local_sum = %lld\n",
           rank, chunk_size, local_sum);

    if (rank != 0) {
        MPI_Send(&local_sum, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    } else {
        long long total_sum = local_sum;
        for (int r = 1; r < size; r++) {
            long long recv_sum;
            MPI_Recv(&recv_sum, 1, MPI_LONG_LONG, r, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_sum += recv_sum;
        }

        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Scatter] Total sum = %lld\n", total_sum);
        printf("[Scatter] Expected  = %lld\n", expected);
        printf("[Scatter] Correct?  = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Scatter] Time      = %.4f sec\n", elapsed);
    }

    free(local_chunk);
    if (rank == 0) free(array);
    MPI_Finalize();
    return 0;
}
