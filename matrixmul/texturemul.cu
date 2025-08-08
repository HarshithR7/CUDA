// designed for optimized 2D spatial access patterns
/*
texture<int,2, cudaReadModeElementType>tex_a; //2- (x,y)- 2d
texture<int,2,cudaReadModeElementType>tex_b; // read data as-is (no normalization)


__global__ void matmul_texture(int *a, int *b, int *c, int ra, int ca, int rb,int cb){

    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int j=blockIdx.y * blockDim.y + threadIdx.y;

    if (i<ra && j<cb){
        int sum=0;
        for (int k=0;k<ca;++k){
        // Fixed indexing: row-major for both matrices
            sum+=tex2D[tex_a,k,i]*tex2D[tex_b,j,k]; // ca has no.of ekements in column of matrix a
        }
        c[i*cb+j]=sum;
    }
}
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

// Modern CUDA kernel using texture objects (not deprecated texture references)
__global__ void matmul_texture(cudaTextureObject_t tex_a, cudaTextureObject_t tex_b, 
                              int *c, int ra, int ca, int rb, int cb) {
    // Calculate global thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row in result matrix
    int j = blockIdx.y * blockDim.y + threadIdx.y; // column in result matrix

    // Bounds checking
    if (i < ra && j < cb) {
        int sum = 0;
        
        // Perform dot product using texture memory
        for (int k = 0; k < ca; ++k) {
            // tex2D<int>(texture_object, x, y) where x=column, y=row
            // For matrix A: we want A[i][k], so x=k, y=i
            // For matrix B: we want B[k][j], so x=j, y=k
            int val_a = tex2D<int>(tex_a, k, i);
            int val_b = tex2D<int>(tex_b, j, k);
            sum += val_a * val_b;
        }
        
        // Store result in row-major order
        c[i * cb + j] = sum;
    }
}

// Host function to set up textures and perform matrix multiplication
void matmul_with_texture(int *h_a, int *h_b, int *h_c, int ra, int ca, int rb, int cb) {
    // Device memory pointers
    cudaArray *d_a_array, *d_b_array;
    int *d_c;
    
    // Create channel descriptors for integer data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
    
    // Allocate CUDA arrays for textures
    cudaMallocArray(&d_a_array, &channelDesc, ca, ra); // width=ca, height=ra
    cudaMallocArray(&d_b_array, &channelDesc, cb, rb); // width=cb, height=rb
    
    // Allocate device memory for result matrix
    cudaMalloc((void**)&d_c, ra * cb * sizeof(int));
    
    // Copy data to CUDA arrays using modern API
    cudaMemcpy2DToArray(d_a_array, 0, 0, h_a, ca * sizeof(int), 
                        ca * sizeof(int), ra, cudaMemcpyHostToDevice);
    cudaMemcpy2DToArray(d_b_array, 0, 0, h_b, cb * sizeof(int), 
                        cb * sizeof(int), rb, cudaMemcpyHostToDevice);
    
    // Create texture objects
    cudaResourceDesc resDesc_a = {};
    resDesc_a.resType = cudaResourceTypeArray;
    resDesc_a.res.array.array = d_a_array;
    
    cudaResourceDesc resDesc_b = {};
    resDesc_b.resType = cudaResourceTypeArray;
    resDesc_b.res.array.array = d_b_array;
    
    // Set texture parameters
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp; // x-direction (columns)
    texDesc.addressMode[1] = cudaAddressModeClamp; // y-direction (rows)
    texDesc.filterMode = cudaFilterModePoint;      // No interpolation
    texDesc.readMode = cudaReadModeElementType;    // Read as integers
    texDesc.normalizedCoords = 0;                  // Use integer coordinates
    
    // Create texture objects
    cudaTextureObject_t tex_a = 0, tex_b = 0;
    cudaCreateTextureObject(&tex_a, &resDesc_a, &texDesc, NULL);
    cudaCreateTextureObject(&tex_b, &resDesc_b, &texDesc, NULL);
    
    // Configure kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((ra + blockSize.x - 1) / blockSize.x, 
                  (cb + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    matmul_texture<<<gridSize, blockSize>>>(tex_a, tex_b, d_c, ra, ca, rb, cb);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, ra * cb * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup texture objects
    cudaDestroyTextureObject(tex_a);
    cudaDestroyTextureObject(tex_b);
    
    // Cleanup arrays and device memory
    cudaFreeArray(d_a_array);
    cudaFreeArray(d_b_array);
    cudaFree(d_c);
}

// Helper function to print matrix
void printMatrix(int *matrix, int rows, int cols, const char* name) {
    std::cout << "\n" << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << "\t";
        }
        std::cout << "\n";
    }
}

// Test function
int main() {
    // Matrix dimensions
    const int ra = 4, ca = 3; // Matrix A: 4x3
    const int rb = 3, cb = 4; // Matrix B: 3x4
    // Result matrix C will be 4x4
    
    // Verify dimensions are compatible
    if (ca != rb) {
        std::cerr << "Error: Incompatible matrix dimensions for multiplication" << std::endl;
        return -1;
    }
    
    // Allocate host memory
    int *h_a = new int[ra * ca];
    int *h_b = new int[rb * cb];
    int *h_c = new int[ra * cb];
    
    // Initialize matrices with test data
    std::cout << "Initializing matrices...\n";
    
    // Matrix A
    for (int i = 0; i < ra; i++) {
        for (int j = 0; j < ca; j++) {
            h_a[i * ca + j] = i + j + 1;
        }
    }
    
    // Matrix B  
    for (int i = 0; i < rb; i++) {
        for (int j = 0; j < cb; j++) {
            h_b[i * cb + j] = (i + 1) * (j + 1);
        }
    }
    
    // Initialize result matrix to zero
    for (int i = 0; i < ra * cb; i++) {
        h_c[i] = 0;
    }
    
    // Print input matrices
    printMatrix(h_a, ra, ca, "Matrix A");
    printMatrix(h_b, rb, cb, "Matrix B");
    
    // Perform matrix multiplication using textures
    std::cout << "\nPerforming matrix multiplication using texture memory...\n";
    matmul_with_texture(h_a, h_b, h_c, ra, ca, rb, cb);
    
    // Print result
    printMatrix(h_c, ra, cb, "Result Matrix C = A × B");
    
    // Verify result with CPU calculation
    std::cout << "\nVerifying result with CPU calculation...\n";
    int *h_c_cpu = new int[ra * cb];
    
    for (int i = 0; i < ra; i++) {
        for (int j = 0; j < cb; j++) {
            int sum = 0;
            for (int k = 0; k < ca; k++) {
                sum += h_a[i * ca + k] * h_b[k * cb + j];
            }
            h_c_cpu[i * cb + j] = sum;
        }
    }
    
    // Compare results
    bool correct = true;
    for (int i = 0; i < ra * cb; i++) {
        if (h_c[i] != h_c_cpu[i]) {
            correct = false;
            break;
        }
    }
    
    if (correct) {
        std::cout << "✓ Results match! Texture-based multiplication is correct.\n";
    } else {
        std::cout << "✗ Results don't match! There's an error in the implementation.\n";
        printMatrix(h_c_cpu, ra, cb, "Expected Result (CPU)");
    }
    
    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_c_cpu;
    
    return 0;
}


/*
Matrix A (4x3):
1       2       3
2       3       4
3       4       5
4       5       6

Matrix B (3x4):
1       2       3       4
2       4       6       8
3       6       9       12

Performing matrix multiplication using texture memory...

Result Matrix C = A × B (4x4):
14      28      42      56
20      40      60      80
26      52      78      104
32      64      96      128

Verifying result with CPU calculation...
✓ Results match! Texture-based multiplication is correct.


*/