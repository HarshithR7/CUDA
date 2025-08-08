#include <cstdio>
#include <cstdlib>



__global__ void c2g(int *in, float *out, int N, int M){
    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int j=blockIdx.y * blockDim.y + threadIdx.y;
    
    if(i<N && j<M){
        int idx=j*N + i;
        int r=in[3*idx];
        int g=in[3*idx+1];
        int b=in[3*idx+2];
        out[idx] = (0.21*r + 0.71*g + 0.07*b);
    }
}

__global__ void intitialize(int *in, int N, int M){
    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int j=blockIdx.y * blockDim.y + threadIdx.y;
    
    if(i<N && j<M){
        int idx=j*N + i;
        in[3 * idx + 0] = (i + j) % 256;         // Red
        in[3 * idx + 1] = (i * 2 + j) % 256;     // Green
        in[3 * idx + 2] = (i + j * 2) % 256;     // Blue
    }
}




// NxM
int main(){

    int *in;
    float *out;
    int N=1024;
    int M=1024;

    cudaMallocManaged(&in,N*M*3*sizeof(int));   
    cudaMallocManaged(&out,N*M*3*sizeof(int));

    dim3 threads(16,16);
    dim3 blocks(M/threads.x,N/threads.y);
    intitialize<<<blocks,threads>>>(in,N,M);
    cudaDeviceSynchronize();

    c2g<<<blocks,threads>>>(in,out,N,M);
    cudaDeviceSynchronize();

    printf("=== Initialized Data ===\n");
    for (int j = 0; j < 5; ++j) {
        for (int i = 0; i < 5; ++i) {
            int idx = j * N + i;
            printf("(%2d,%2d) r: %2d g:%2d b:%2d, out: %f\n", i, j, in[3*idx], in[3*idx+1], in[3*idx+2], out[idx]);
        }
    }



    cudaFree(in);
    cudaFree(out);

    return 0;
}


/*
=== Initialized Data ===
( 0, 0) r:  0 g: 0 b: 0, out:  0
( 1, 0) r:  1 g: 2 b: 1, out:  1
( 2, 0) r:  2 g: 4 b: 2, out:  3
( 3, 0) r:  3 g: 6 b: 3, out:  5
( 4, 0) r:  4 g: 8 b: 4, out:  6
( 0, 1) r:  1 g: 1 b: 2, out:  1
( 1, 1) r:  2 g: 3 b: 3, out:  2
( 2, 1) r:  3 g: 5 b: 4, out:  4
( 3, 1) r:  4 g: 7 b: 5, out:  6
( 4, 1) r:  5 g: 9 b: 6, out:  7
( 0, 2) r:  2 g: 2 b: 4, out:  2
( 1, 2) r:  3 g: 4 b: 5, out:  3
( 2, 2) r:  4 g: 6 b: 6, out:  5
( 3, 2) r:  5 g: 8 b: 7, out:  7
( 4, 2) r:  6 g:10 b: 8, out:  8
( 0, 3) r:  3 g: 3 b: 6, out:  3
( 1, 3) r:  4 g: 5 b: 7, out:  4
( 2, 3) r:  5 g: 7 b: 8, out:  6
( 3, 3) r:  6 g: 9 b: 9, out:  8
( 4, 3) r:  7 g:11 b:10, out:  9
( 0, 4) r:  4 g: 4 b: 8, out:  4
( 1, 4) r:  5 g: 6 b: 9, out:  5
( 2, 4) r:  6 g: 8 b:10, out:  7
( 3, 4) r:  7 g:10 b:11, out:  9
( 4, 4) r:  8 g:12 b:12, out: 11
*/
