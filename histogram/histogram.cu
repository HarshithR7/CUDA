
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void histogram_unsaturated (uint32_t* input, uint32_t* bins,uint32_t len){

    int i=blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i<len){

        uint32_t val=input[i];
    //    bins[val]+=1; // Multiple threads may try to increment the same bin at the same time → data race. use atomicAdd() when multiple threads write to the same memory location.
        atomicAdd(&bins[val],1);
    }

}

__global__ void saturation(uint32_t* input, uint32_t* bins,uint32_t len, uint32_t num_bins){
    int i=blockDim.x * blockIdx.x + threadIdx.x;
     if(i<num_bins){
        if(bins[i] > 127){
            printf("bins[%d]: %u\n",i,bins[i]);
            bins[i] = 127;
        }
    }
}


__global__ void initialize_data(uint32_t* bins, uint32_t num_bins){
    int i=blockDim.x * blockIdx.x + threadIdx.x;
    
    if(i<num_bins){
        bins[i]=0;
    }

}
// #define num_bins 4096   - preprocessor macro,
// #define len 65536
int main(){
    uint32_t* bins;
    uint32_t* input;
    uint32_t len=65536;
    uint32_t num_bins=4096;

    cudaMallocManaged(&bins,num_bins * sizeof(uint32_t));
    cudaMallocManaged(&input,len * sizeof(uint32_t));

    for (uint32_t i=0;i<len;i++){
        input[i]=rand()%4096;
    }

    int threads=128; // max limmit is 1024
    int blocks_input=(len+threads-1)/threads;
    int blocks_bin= (num_bins+threads-1)/threads;


    initialize_data<<<blocks_bin,threads>>>(bins,num_bins);
    cudaDeviceSynchronize();

    histogram_unsaturated<<<blocks_input,threads>>>(input,bins,len);
    cudaDeviceSynchronize();

    saturation<<<blocks_input,threads>>>(input,bins,len,num_bins);
    cudaDeviceSynchronize();

    for(int i=0;i<250;++i){

        printf("bins[%d]: %u\n",i,bins[i]);
    }


    // ---- CPU reference histogram for verification ----
    uint32_t* cpu_bins = (uint32_t*)calloc(num_bins, sizeof(uint32_t));
    for (uint32_t i = 0; i < len; ++i) {
        cpu_bins[input[i]]++;
    }   

    // ---- Compare CPU and GPU histograms ----
    //bool correct = true;
    for (uint32_t i = 0; i < num_bins; ++i) {
        if (cpu_bins[i] != bins[i]) {
            printf("Mismatch at bin[%u]: CPU = %u, GPU = %u\n", i, cpu_bins[i], bins[i]);
           // correct = false;
            break;  // Or remove break to see all mismatches
        }
    }

    free(cpu_bins);
    cudaFree(input);
    cudaFree(bins);

    return 0;


}

/*
want to use these values in device kernels, always use:

const uint32_t len = 65536;
const uint32_t num_bins = 4096;

or

constexpr uint32_t len = 65536;
constexpr uint32_t num_bins = 4096;
*/