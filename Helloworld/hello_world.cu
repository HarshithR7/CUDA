#include <cstdio> // #include this is a preprocessor directive that tells compiler to include the contents of a file before compilation begins
// cstdio includes the C++ standard input/output library

__global__ void mykernel(void) {  // global is a cuda keyword that tells mykernal function to run on GPU, it returns nothing, it takes no arguments
}

int main() { // it returns integer 1-failure 0-success, main is the entry point of c++ program
  
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  
  
  mykernel<<<1,1>>>(); // calling mykernal GPU function on 1  block and 1 thread per block. launched 1 thread in 1 block
  cudaEventRecord(start);
  printf("Hello World!\n");
  cudaEventRecord(stop);
  cudaDeviceSynchronize();
  float ms=0;
  cudaEventElapsedTime(&ms,start,stop);
  printf("Kernal Execution time: %f ms \n",ms);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}

// CPu is faster fro printing jobs