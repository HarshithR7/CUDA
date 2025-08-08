#include <cstdio> // gives print commands
#include <cstdlib> // gives exit() command

// check for errors,  cudaGetErrorString converts the error code into readable string, // exit- exits the program with an error
#define cudaErrorCheck() {                                                              \
        cudaError_t e=cudaGetLastError();                                                   \
        if (e!=cudaSuccess){                                                                \
            printf("cuda failure %s:%d: '%s'\n", __FILE__,__LINE__,cudaGetErrorString(e));  \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    }


__global__ void kernel(int *a){

int i=blockIdx.x * blockDim.x + threadIdx.x;
a[i]=i; // each thread writes it index to the array
}

int main(){
    int N=4097;
    int threads=128;
    int blocks=(N+threads-1)/threads;
    int *a;

    cudaMallocManaged(&a,N*sizeof(int));// allocates unified memory
    kernel<<<blocks,threads>>>(a); 
    cudaDeviceSynchronize();
    for (int i=0;i<10;i++)
        printf("%d\n",a[i]);
    cudaFree(a);

    cudaErrorCheck();
    return 0;


}
    


