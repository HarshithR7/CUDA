// ASCII characters - 128

#include <cstdio>
#include <cstdlib>
#include <cstdint>

__global__ void ascii_hist(char *input_d,uint32_t *bins_d, int num_bins,int len){

    int i = blockIdx.x * blockDim.x + threadIdx.x; // row
    int tid=threadIdx.x;

    __shared__ uint32_t local_bins[128]; //In CUDA, shared memory arrays must have their size known at compile time.
    // or extern __shared__ uint32_t local_bins[]; // dynamic memory sharing


    if(tid<num_bins){
        local_bins[tid]=0;
    }

   if(i<len){
        uint32_t val=input_d[i];

        atomicAdd(&local_bins[val],1);

    }

    if(tid<num_bins){

        atomicAdd(&bins_d[tid],local_bins[tid]);
    }

}


//#define num_bins 128

int main(){
    int num_bins=128;
    int len=655536;

    char *input= (char*)malloc(len*sizeof(char));
    //uint32_t *bins= (uint32_t*)malloc(len*sizeof(uint32_t)); // allocate memory but don't initialize it
    uint32_t *bins= (uint32_t*)calloc(num_bins,sizeof(uint32_t)); // allocates and initializes to zero;

    for (int i=0;i<len;i++){

        input[i] = rand()%num_bins;

    }

    char *input_d;
    uint32_t *bins_d;

    cudaMalloc((void**)&input_d,len*sizeof(char));
    cudaMalloc((void**)&bins_d,num_bins*sizeof(uint32_t));

    cudaMemcpy(input_d,input,len*sizeof(char),cudaMemcpyHostToDevice);
    cudaMemcpy(bins_d,bins,num_bins*sizeof(uint32_t),cudaMemcpyHostToDevice);

    int threads=256;
    int blocks_input=(len+threads-1)/threads;
    //int blocks_bin=(num_bins+threads-1)/threads;


    ascii_hist<<<blocks_input,threads>>>(input_d,bins_d, num_bins,len );
    cudaDeviceSynchronize();

    cudaMemcpy(bins,bins_d,num_bins*sizeof(uint32_t), cudaMemcpyDeviceToHost);

    
    for(int i=0;i<129;++i){
        printf("bins[%d]:%u\n",i,bins[i]);
    }


    cudaFree(input_d);
    cudaFree(bins_d);
    free(bins);
    free(input);

    return 0;


    
}