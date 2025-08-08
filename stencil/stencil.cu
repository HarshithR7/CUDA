
//i = row = Y axis → height

//j = column = X axis → width

// k = depth = Z axis

#include <cstdio> // input/output functions (printf, fprintf)
#include <cstdlib> // library functions (malloc, free, rand)
#include <cuda_runtime.h> // CUDA runtime API functions (cudaMalloc, cudaMemcpy, etc.)

// 3D array indexing macro - converts 3D coordinates (i,j,k) to 1D linear index
// Formula: index = ((i * width) + j) * depth + k
// This linearizes 3D memory layout for GPU global memory access
#define value(array, i, j, k, width, depth) array[((i) * width + (j)) * depth + (k)]
#define in(i, j, k) value(input_d, i, j, k, width, depth)
#define out(i, j, k) value(output_d, i, j, k, width, depth)

__device__ float clamp_val(float val, float start,float end){

    return fmaxf(fminf(val,end),start);// fmaxf/fminf - GPU-optimized single-precision floating point min/max functions
    // These are faster than regular min/max on GPU hardware
}

__global__ void stencil7(float *input_d,float *output_d,int width, int height, int depth){

    const int tile_width=8;
    const int tile_depth=8;
    const int tile_height=8;
    const int radius=1;
    __shared__ float tile[tile_width+2][tile_height+2][tile_depth+2];

    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int j=blockIdx.y * blockDim.y + threadIdx.y;
    int k=blockIdx.z * blockDim.z + threadIdx.z;

    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int tz=threadIdx.z;

    int shared_x=tx+radius;
    int shared_y=ty+radius;
    int shared_z=tz+radius;
    

    //load data into share memory

    if(i<height && j< width && k<depth){
        tile[shared_x][shared_y][shared_z]= in(i,j,k);
    }else{

        tile[shared_x][shared_y][shared_z]= 0.0f;
    }
//x-1
    if(tx == 0 && i > 0){
            tile[0][shared_y][shared_z]=in(i-1,j,k);
        } else if(tx==0){
            tile[0][shared_y][shared_z] = 0.0f;
        }
    

    // x+1 (right)
    if(tx == blockDim.x - 1 && i + 1 < height){
        tile[shared_x + 1][shared_y][shared_z] = in(i+1, j, k);
    } else if(tx == blockDim.x - 1) {
        tile[shared_x + 1][shared_y][shared_z] = 0.0f;
    }

    // y-1 (top)
    if(ty == 0 && j > 0){
        tile[shared_x][0][shared_z] = in(i, j-1, k);
    } else if(ty == 0) {
        tile[shared_x][0][shared_z] = 0.0f;
    }

    // y+1 (bottom)
    if(ty == blockDim.y - 1 && j + 1 < width){
        tile[shared_x][shared_y + 1][shared_z] = in(i, j+1, k);
    } else if(ty == blockDim.y - 1) {
        tile[shared_x][shared_y + 1][shared_z] = 0.0f;
    }

    // z-1 (front)
    if(tz == 0 && k > 0){
        tile[shared_x][shared_y][0] = in(i, j, k-1);
    } else if(tz == 0) {
        tile[shared_x][shared_y][0] = 0.0f;
    }

    // z+1 (back)
    if(tz == blockDim.z - 1 && k + 1 < depth){
        tile[shared_x][shared_y][shared_z + 1] = in(i, j, k+1);
    } else if(tz == blockDim.z - 1) {
        tile[shared_x][shared_y][shared_z + 1] = 0.0f;
    }
    __syncthreads(); // wait for all threads
        // Compute stencil for all valid points (including boundary)

    // Compute stencil for all valid points (including boundary)
    if (i < height && j < width && k < depth) {
        float result;
        
        // For interior points, compute full stencil
        if (i > 0 && i < height - 1 && j > 0 && j < width - 1 && k > 0 && k < depth - 1) {
            result = 
                tile[shared_x - 1][shared_y][shared_z] +     // left (i-1)
                tile[shared_x + 1][shared_y][shared_z] +     // right (i+1)
                tile[shared_x][shared_y - 1][shared_z] +     // top (j-1)
                tile[shared_x][shared_y + 1][shared_z] +     // bottom (j+1)
                tile[shared_x][shared_y][shared_z - 1] +     // front (k-1)
                tile[shared_x][shared_y][shared_z + 1] -     // back (k+1)
                6.0f * tile[shared_x][shared_y][shared_z];   // center
        } else {
            // For boundary points, just copy input (or apply different boundary condition)
            result = tile[shared_x][shared_y][shared_z];
        }
        
        out(i, j, k) = clamp_val(result, 0.0f, 255.0f);
    }
} 


int main(){

    int depth=64;
    int width=64;
    int height=64;

    size_t size=depth*height*width*sizeof(float);

    float *input = (float*)malloc(size);
    float *output = (float*)malloc(size);

    //initialize data
    for(int i=0;i<depth*height*width;++i){
        input[i]=(float)(rand()%256);
    }
    memset(output, 0, size);

    float *input_d,*output_d;
    cudaMalloc((void**)&input_d, size);
    cudaMalloc((void **)&output_d, size);
    cudaMemcpy(input_d,input,size,cudaMemcpyHostToDevice);
    cudaMemcpy(output_d, output, size, cudaMemcpyHostToDevice);
    dim3 threads(8, 8, 8);
    dim3 blocks((height + threads.x - 1) / threads.x, 
                (width + threads.y - 1) / threads.y, 
                (depth + threads.z - 1) / threads.z);

    stencil7<<<blocks,threads>>>(input_d,output_d,width,height,depth);
    cudaDeviceSynchronize();

    cudaMemcpy(output, output_d, size, cudaMemcpyDeviceToHost);

    // Print some output to verify
    for (int i = 0; i < 100; ++i) {
        printf("output[%d] = %.2f\n", i, output[i]);
    }
    cudaFree(input_d);
    cudaFree(output_d);
    free(input);
    free(output);
    return 0;
}
    
