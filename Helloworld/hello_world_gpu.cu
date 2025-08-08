#include <cstdio> // #include this is a preprocessor directive that tells compiler to include the contents of a file before compilation begins
// cstdio includes the C++ standard input/output library

__global__ void mykernel(void) {  // global is a cuda keyword that tells mykernal function to run on GPU, it returns nothing, it takes no arguments
printf("Hello World!\n");
}

int main() { // it returns integer 1-failure 0-success, main is the entry point of c++ program
  cudaEvent_t start,stop; // defined 2 variables
  cudaEventCreate(&start); // allocated and initialized them in gpu structure
  cudaEventCreate(&stop);
  cudaEventRecord(start); // mark time before kernal

  mykernel<<<1,1>>>(); // calling mykernal GPU function on 1  block and 1 thread per block. launched 1 thread in 1 block
  cudaEventRecord(stop); // mark time after launch

  cudaDeviceSynchronize(); // without this int main() program would exit without waiitng for mykernal response,(response from mykernal not capture)
  
  float milliseconds=0;
  cudaEventElapsedTime(&milliseconds,start,stop);

  printf("Kernal Execution time:%f ms\n", milliseconds);
  cudaEventDestroy(start); // free the resources
  cudaEventDestroy(stop);
  return 0;
}

// <<<1,5>>> Kernal Execution time:14.382912 ms
// <<<5,1>>> Kernal Execution time:17.272833 ms

// <<<1,6>>> Kernal Execution time:16.254881 ms
// <<<2,3>>> Kernal Execution time:18.817024 ms
// <<<3,2>>> Kernal Execution time:17.187840 ms
// <<<6,1>>> Kernal Execution time:16.132095 ms


/*
Why Kernel Execution Time Changes Every Run
Here are the main reasons:

1. GPU Warm-Up / Context Initialization
The first run of a CUDA program creates a context and loads it into the GPU driver.

Subsequent runs may reuse this, or the context may be reloaded.

This cold start or warm start causes time variations.

📌 Tip: Ignore the first run when benchmarking.

2. Thread Scheduling on GPU (SM Load)
CUDA threads are mapped to Streaming Multiprocessors (SMs).

If another process (e.g. a background graphical application) is using GPU resources, your threads might be delayed or scheduled differently.

⚠️ If you're on a shared or GUI system (especially laptop GPUs), this matters more.

3. Clock Frequency / Thermal Throttling
Modern GPUs use dynamic clock scaling (like CPUs).

If the GPU is too hot, it may throttle down.

If idle, it may clock up suddenly when you run your code.

4. Asynchronous Execution Behavior
CUDA kernels are asynchronous, so if cudaDeviceSynchronize() timing isn't placed carefully, background delays can sneak in.

Even with cudaEventRecord(), the precision is microsecond-level — enough to pick up tiny differences.

5. Operating System & Driver Jitter
The OS may interrupt your program slightly with:

Background tasks

Driver-level work

Memory paging or cache behavior

6. Non-Deterministic Memory Access / L1/L2 Cache Hits
If your kernel accesses global memory or shared memory, cache behavior can introduce small time shifts due to hits/misses.

*/