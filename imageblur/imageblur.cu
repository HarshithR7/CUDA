#include <cstdio>
#define blursize 3
#define channels 3
//MxN - idx=N*i+j
__global__ void blur(int *in, int *out,int M,int N){
    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int j=blockIdx.y * blockDim.y + threadIdx.y;
    for (int c=0;c<channels;++c)
    {
        if(i<N && j<M){  // M is height N is width
            int idx=i*M + j;
            
            
            float pixval=0;
            float pixels=0;
            for (int blurrow=-blursize;blurrow<blursize+1;++blurrow){
                for (int blurcol=-blursize;blurcol<blursize+1;++blurcol){

                    int currow= blurrow+j;
                    int curcol=blurcol+i;

                    if (currow>-1 && currow<M && curcol>-1 &&curcol<N){

                        pixval+=in[channels* (curcol*M + currow) + c];
                        pixels++;
                    }
                }

        
            }
      
        out[idx* channels + c]=(pixval/pixels);
        }
       
    }

}

__global__ void intitialize(int *in, int M, int N){
    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int j=blockIdx.y * blockDim.y + threadIdx.y;
    
    if(i<N && j<M){
        int idx=j*N + i;
        in[3 * idx + 0] = (i + j) % 256;         // Red
        in[3 * idx + 1] = (i * 2 + j) % 256;     // Green
        in[3 * idx + 2] = (i + j * 2) % 256;     // Blue
    }
}

int main(){

    int *in;
    int *out;
 //   int blursize=3;
    int N=1024;
    int M=1024;

    dim3 threads(16,16);
    dim3 blocks(N/threads.x,M/threads.y);

    cudaMallocManaged(&in, N*M*3*sizeof(int)); 
    cudaMallocManaged(&out, N*M*3*sizeof(int));

    intitialize<<<blocks,threads>>>(in,M,N);
    cudaDeviceSynchronize();

    blur<<<blocks,threads>>>(in,out,M,N);
    cudaDeviceSynchronize();

    printf("=== Data ===\n");
    //for (int c=0;c<channels;++c){
        for (int j = 0; j < 5; ++j) {
            for (int i = 0; i < 5; ++i) {
                int idx = j * N + i;
                printf("(%2d,%2d) r: %2d g:%2d b:%2d, out_r: %2d, out_g: %2d, out_b: %2d\n", i, j, in[3*idx], in[3*idx+1], in[3*idx+2], out[3*idx], out[3*idx+1], out[3*idx+2]);
            }
        }
   // }


    cudaFree(in);
    cudaFree(out);

    return 0;

}