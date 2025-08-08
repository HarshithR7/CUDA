#include <cstdio> 
#include <cstdlib> // provides malloc, rand, exit
#include <iostream> // for printing to the terminal std::cout and std::cerr
#include <cuda_runtime.h> // for cuda runtime api functions like cudamalloc, cudamemcpy

__global__ void vector_add(int *a, int *b, int *c,int N) {

    int i= blockDim.x * blockIdx.x + threadIdx.x;
    if (i<N){
        c[i] = a[i] + b[i];
    }
}

__global__ void initialize_data(int *a, int *b, int N){
    int i= blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i<N){
        a[i] = i;
        b[i] = 2*i;
    }

}

int main () {
    
    int N,
    int threads=32;
    int blocks= (N+threads-1)/N;

    // host memory ptr = (type*)malloc(size); free(ptr);
    int *ha = (int*)malloc(N*sizeof(int)); // int* means using malloc[memory allocation] for integer storage, pointer ha has the address of memory allocated.
    int *hb = (int*)malloc(N*sizeof(int));
    int *hc = (int*)malloc(N*sizeof(int));
     
    //device memory cudaMalloc((void**)&d_ptr, size); cudaFree(d_ptr);
    int *da, *db, *dc;
    cudaMalloc((void**)&da,N*sizeof(int));
    cudaMalloc((void**)&db,N*sizeof(int));
    cudaMalloc((void**)&dc,N*sizeof(int));

    // copy from host to device
    cudaMemcpy(da,ha,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,N*sizeof(int),cudaMemcpyHostToDevice);

    // intitalize data and perform additon
    initialize_data<<<blocks,threads>>>(da,db,N);
    vector_add<<<blocks,threads>>>(da,db,dc,N);
    cudaDeviceSynchronize();
    
    // copy result back to host
    cudaMemcpy(hc,dc,N*sizeof(int),cudaMemcpyDeviceToHost);

    bool success=true;
    for (i=0:i<N:++i){
        if(hc[i]!= ha[i]+hb[i]){
                std::cerr << "mismatch at index" << i <<":" <<"hc[i]"<<"!=" <<ha[i]+hb[i] <<std::cendl;
                success=false;
                break;
        }
    }

    std::cout<< (success? "vector additon successful ": "failed operation") << std::endl;
    free(ha);
    free(hb);
    free(hc);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return 0;
}


/*
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

*/