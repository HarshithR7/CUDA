#include <cstdio>
#include <cstdlib>
#include <cstdint>


__constant__ uint32_t const_mem[16384];

__global__ void hist(uint32_t *bins_d, int len, int num_bins){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row
    if(i < len){
        uint32_t val = const_mem[i];
        atomicAdd(&bins_d[val],1);
    }
}



int main(){
    int len=16384;
    int num_bins=4096;
    uint32_t *input=(uint32_t*)malloc(len*sizeof(uint32_t));
    uint32_t *bins=(uint32_t*)malloc(num_bins*sizeof(uint32_t));
    uint32_t *bins_d;

    //cudaMalloc((void**)&input_d,len*sizeof(uint32_t)); // cuda malloc is generic allocator &input_d is uint32_t** type, casting to void**
    cudaMalloc((void**)&bins_d,num_bins*sizeof(uint32_t));   // for global memory

    for (int i=0;i<len;++i){
        input[i]=rand()% num_bins;
    }

    for (int i = 0; i < num_bins; ++i) // here i++,++i work same because result of expression isn't used.
    {
        bins[i]=0;
    }
    cudaMemcpy(bins_d,bins,num_bins*sizeof(uint32_t),cudaMemcpyHostToDevice);
    int threads= 256;
    int blocks=(len+threads-1)/threads;

    
    cudaMemcpyToSymbol(const_mem,input,len*sizeof(uint32_t));
    hist<<<blocks,threads>>>(bins_d,len,num_bins);
    cudaDeviceSynchronize();

    cudaMemcpy(bins,bins_d,num_bins*sizeof(uint32_t),cudaMemcpyDeviceToHost);



    for(int i=0;i<100;++i){
        printf("bins[%d]:%u\n",i,bins[i]);
    }

    cudaFree(bins_d);
    cudaFree(const_mem);
    free(bins);
    free(input);

    return 0;
  
}