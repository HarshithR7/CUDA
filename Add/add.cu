//sudo /opt/nvidia/nsight-compute/2024.2.0/target/linux-desktop-glibc_2_11_3-x64/ncu ./Add/add
/*
harshithr7@Harshith:/mnt/c/CUDA$ which ncu
/opt/nvidia/nsight-compute/2024.2.0/target/linux-desktop-glibc_2_11_3-x64/ncu
harshithr7@Harshith:/mnt/c/CUDA$*/
#include <cstdio>
#include <cstdlib>

#define cudaErrorCheck() { \
        cudaError_t e=cudaGetLastError(); \
        if (e!=cudaSuccess) {\
            printf("cuda failure %s:%d:'%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
            exit(EXIT_FAILURE); \
        } \
    }

    // row major addressing . access is like a[0], a[4], a[8], .....
__global__ void add_v1(int *a, int *b, int *c, int N, int M) {

    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int j=blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i<N && j<M) {
        int idx=i*M+j;

        c[idx]=a[idx]+b[idx];
    }
}


// GPUs access memory in chunks, and they're most efficient when consecutive threads access consecutive memory addresses 
// (called coalesced memory access).

// column major addressing. access is like a[0], a[1], a[2], .....
__global__ void add_v2(int *a, int *b, int *c, int N, int M) {

    int j=blockIdx.x * blockDim.x + threadIdx.x;
    int i=blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i<N && j<M) {
        int idx=i*N+j;

        c[idx]=a[idx]+b[idx];
    }
}

__global__ void initialize_data(int *a, int *b, int N, int M){

    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int j=blockIdx.y * blockDim.y + threadIdx.y;
    if (i<N && j<M) {
        int idx=i*M+j;
        a[idx]=idx;
        b[idx]=2*idx;
    }
}

__global__ void verify_data(int *c1, int *c2, int N, int M, int *error){

    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int j=blockIdx.y * blockDim.y + threadIdx.y;
    if (i<N && j<M) {
        int idx=i*M+j;
        if(c1[idx]!=c2[idx]){
            printf("error is at %d in c1:%d and c2:%d",idx,c1[idx],c2[idx]);
            *error=1;
        }
    }
}

int main(){


    int N=1024;
    int M=1024;
    int *a, *b, *c1,*c2;
    dim3 threads(32,32);
    dim3 blocks(N/threads.x,M/threads.y);
    
    
    cudaMallocManaged(&a,N*M*sizeof(int));
    cudaMallocManaged(&b,N*M*sizeof(int));
    cudaMallocManaged(&c1,N*M*sizeof(int));
    cudaMallocManaged(&c2,N*M*sizeof(int));
    
    // intialize_data(a,b,N,M); it will be cpu initalization when write void intialization_data function

    initialize_data<<<blocks,threads>>>(a,b,N,M);

    cudaEvent_t v1_start,v1_end;
    cudaEventCreate(&v1_start);
    cudaEventCreate(&v1_end);

    cudaEventRecord(v1_start);
    add_v1<<<blocks,threads>>>(a,b,c1,N,M);
    cudaEventRecord(v1_end);
    cudaDeviceSynchronize();
    float v1_time=0;   
    cudaEventElapsedTime(&v1_time,v1_start,v1_end);


    cudaEvent_t v2_start,v2_end;
    cudaEventCreate(&v2_start);
    cudaEventCreate(&v2_end);

    cudaEventRecord(v2_start);
    add_v2<<<blocks,threads>>>(a,b,c2,N,M);
    cudaEventRecord(v2_end);
    cudaDeviceSynchronize();
    float v2_time=0;
    cudaEventElapsedTime(&v2_time,v2_start,v2_end);

    printf("Execution time Add_v1: %f, add_v2: %f \n",v1_time,v2_time );
    // Execution time Add_v1: 21.011456, add_v2: 0.904192 
    cudaEventDestroy(v1_start);
    cudaEventDestroy(v1_end);
    cudaEventDestroy(v2_start);
    cudaEventDestroy(v2_end);

    int *error;
    cudaMallocManaged(&error,0,sizeof(int));
    verify_data<<<blocks,threads>>>(c1,c2,N,M,error);
    cudaDeviceSynchronize();


    cudaErrorCheck();

    cudaFree(a);
    cudaFree(b);
    cudaFree(c1);
    cudaFree(c2);

  
    return 0;

}


/*
/// for average execution time
float total_v1=0;
for(int i=0:i<100:++i){
    cudaEventRecord(v2_start);
    add_v2<<<blocks,threads>>>(a,b,c2,N,M);
    cudaEventRecord(v2_end);
    cudaDeviceSynchronize();
    float v2_time=0;
    cudaEventElapsedTime(&v2_time,v2_start,v2_end);

    total_v1+=ms;

    }
*/