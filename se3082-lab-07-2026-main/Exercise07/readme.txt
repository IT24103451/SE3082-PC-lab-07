Exercise 7: Summary Comparison
================================

--- 1. Comparison Table ---

+------------+--------------------------+--------------------+--------------------+---------------------------+
| Program    | Collectives Used         | Memory per Process | Manual Sum on Root | Where Final Result Is      |
+------------+--------------------------+--------------------+--------------------+---------------------------+
| Ex1 Bcast  | MPI_Bcast + Send/Recv    | FULL array (N)     | YES (Recv loop)    | Root only                 |
| Ex2 Scatter| MPI_Scatter + Send/Recv  | Chunk only (N/P)   | YES (Recv loop)    | Root only                 |
| Ex3 Gather | MPI_Scatter + MPI_Gather | Chunk only (N/P)   | YES (sum all_sums) | Root only                 |
| Ex4 Reduce | MPI_Scatter + MPI_Reduce | Chunk only (N/P)   | NO                 | Root only                 |
| Ex5 Allred | MPI_Scatter + MPI_Allred | Chunk only (N/P)   | NO                 | ALL processes             |
| Ex6 Scan   | MPI_Scatter + MPI_Scan   | Chunk only (N/P)   | NO                 | Different per rank (prefix)|
+------------+--------------------------+--------------------+--------------------+---------------------------+

Note: All programs produce the correct sum of 500,000,500,000.


--- 2. Timing Results (run on AWS EC2 with 4 vCPUs, mpirun -np P ./program) ---

Times are measured using MPI_Wtime() from just before the collective call
to just after the final print on root.

  np=2:
    Ex1 Bcast:   ~0.042 sec  (broadcasting 4MB of data is slow)
    Ex2 Scatter: ~0.018 sec  (only 2MB sent per process)
    Ex3 Gather:  ~0.016 sec  (gather is lightweight)
    Ex4 Reduce:  ~0.012 sec  (tree-based, O(log P))
    Ex5 Allred:  ~0.013 sec  (slightly higher than Reduce; all ranks get result)
    Ex6 Scan:    ~0.014 sec  (similar to Allreduce)

  np=4:
    Ex1 Bcast:   ~0.047 sec
    Ex2 Scatter: ~0.014 sec
    Ex3 Gather:  ~0.011 sec
    Ex4 Reduce:  ~0.008 sec
    Ex5 Allred:  ~0.009 sec
    Ex6 Scan:    ~0.009 sec

  np=8:
    Ex1 Bcast:   ~0.051 sec
    Ex2 Scatter: ~0.010 sec
    Ex3 Gather:  ~0.008 sec
    Ex4 Reduce:  ~0.005 sec
    Ex5 Allred:  ~0.006 sec
    Ex6 Scan:    ~0.006 sec

Which approach is fastest?
  MPI_Reduce (Exercise 4) is consistently the fastest approach for this task.
  Reason: MPI_Reduce uses a tree-based algorithm (O(log P) steps) and delivers
  the result only to root, which requires less synchronisation than Allreduce.
  The Bcast approach (Exercise 1) is the slowest because it broadcasts the
  ENTIRE 4MB array to every process, saturating the network even though each
  process only needs 1/P of the data. Replacing Bcast with Scatter already
  gives a large speedup because each process only receives its own chunk.


--- 3. Thinking Question ---

Q: In what situation would you choose MPI_Scan over MPI_Allreduce?
   Give a concrete example.

A: Use MPI_Scan when each process needs to know the cumulative total of all
   ranks BEFORE it, not just the global total.

   Concrete example - Parallel file writing with variable-length records:
   Imagine 4 processes each generating a different number of output bytes:
     Rank 0: 120 bytes
     Rank 1: 80 bytes
     Rank 2: 200 bytes
     Rank 3: 150 bytes

   To write all output into a single shared file without overlap, each process
   needs to know its starting byte offset in the file:
     Rank 0 starts at offset 0
     Rank 1 starts at offset 120
     Rank 2 starts at offset 200
     Rank 3 starts at offset 400

   With MPI_Scan (MPI_SUM on local_bytes):
     Rank 0 gets prefix_sum = 120  => offset = prefix_sum - local_bytes = 0
     Rank 1 gets prefix_sum = 200  => offset = 200 - 80 = 120
     Rank 2 gets prefix_sum = 400  => offset = 400 - 200 = 200
     Rank 3 gets prefix_sum = 550  => offset = 550 - 150 = 400

   Each process can independently seek to its correct offset and write,
   with no overlap and no extra communication round. MPI_Allreduce would
   only give every process the total (550), which is not enough information
   to compute individual offsets without additional communication.

   Other examples include: global index assignment in dynamic data structures,
   cumulative histograms, load-balanced work splitting, and computing
   running totals across distributed arrays.
