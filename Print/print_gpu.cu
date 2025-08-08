#include <cstdio>
#include <cstdlib>



#define cudaErrorCheck() { \
    cudaError_t e=cudaGetLastError(); \
    if (e!=cudaSuccess){ \
        printf("cuda failure %s:%d: %s\n", __FILE__,__LINE__,cudaGetErrorString(e));\
        exit(EXIT_FAILURE); \
    }   \
}

__global__ void kernel(int *a) {
    int i=blockIdx.x * blockDim.x + threadIdx.x;
    a[i]=i;
    printf("values of a:%d\n", a[i]);

}




int main(){
    int N=40;
    int threads=12;
    int blocks=(N+threads-1)/threads;
    int *a;
    cudaMallocManaged(&a, N*sizeof(int));
    kernel<<<blocks,threads>>>(a);
    cudaDeviceSynchronize();

    cudaFree(a);
    cudaErrorCheck();
    return 0;

}
