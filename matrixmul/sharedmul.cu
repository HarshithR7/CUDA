


/*

#define tilesize 16;
__global__ void matmul_shared(int *a, int *b, int *c, int ra, int ca, int rb,int cb){
    
    __shared__ int tile_a[tilesize][tilesize];
    __shared__ int tile_b[tilesize][tilesize];

    int i=blockIdx.x * tilesize + threadIdx.x; // column
    int j=blockIdx.y * tilesize + threadIdx.y; //row

    int sum=0;

    for(int tile=0; tile<(ca+tilesize-1)/tilesize;++tile){  //looping over tiles
        int a_col = tile*tilesize+threadIdx.x; // which column
        if (j<ra && (a_col )<ca){ // matrix a - load horizontal slices, same row and different columns
            tile_a[threadIdx.y][threadIdx.x] = a[j*ca + a_col ];
        }
        else{
            tile_a[threadIdx.y][threadIdx.x] =0;
        }
        
        int b_row = tile * TILE_SIZE + threadIdx.y;

        if((b_row)<rb && i<cb){  // load verticale slices of B, different rows same columns

            tile_b[threadIdx.y][threadIdx.x] = b[(b_row)*cb+i];
        }
        else{
            tile_b[threadIdx.y][threadIdx.x] =0;
        }  
         
        __syncthreads(); // Wait for all threads to load data
        
        // Compute partial sum using shared memory
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        
        __syncthreads(); // Wait before loading next tile
    }
    // Write result
    if (j < ra && i < cb) {
        c[j * cb + i] = sum;
    }
}

*/



#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define TILE_SIZE 16

__global__ void matmul_shared(int *a, int *b, int *c, int ra, int ca, int rb, int cb) {
    
    // Shared memory tiles for matrices A and B
    __shared__ int tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ int tile_b[TILE_SIZE][TILE_SIZE];

    // Calculate global thread indices
    int i = blockIdx.x * TILE_SIZE + threadIdx.x; // column in result matrix
    int j = blockIdx.y * TILE_SIZE + threadIdx.y; // row in result matrix

    int sum = 0;

    // Loop over tiles needed to compute one output element
    for (int tile = 0; tile < (ca + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        
        // Load tile from matrix A (horizontal slice: same row, different columns)
        int a_col = tile * TILE_SIZE + threadIdx.x;
        if (j < ra && a_col < ca) {
            tile_a[threadIdx.y][threadIdx.x] = a[j * ca + a_col];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        
        // Load tile from matrix B (vertical slice: different rows, same column)
        int b_row = tile * TILE_SIZE + threadIdx.y;
        if (b_row < rb && i < cb) {
            tile_b[threadIdx.y][threadIdx.x] = b[b_row * cb + i];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }
         
        __syncthreads(); // Wait for all threads to load data
        
        // Compute partial sum using shared memory tiles
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        
        __syncthreads(); // Wait before loading next tile
    }
    
    // Write result to global memory
    if (j < ra && i < cb) {
        c[j * cb + i] = sum;
    }
}

// Naive matrix multiplication for comparison
__global__ void matmul_naive(int *a, int *b, int *c, int ra, int ca, int rb, int cb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // column
    int j = blockIdx.y * blockDim.y + threadIdx.y; // row
    
    if (j < ra && i < cb) {
        int sum = 0;
        for (int k = 0; k < ca; ++k) {
            sum += a[j * ca + k] * b[k * cb + i];
        }
        c[j * cb + i] = sum;
    }
}

// Host matrix multiplication for verification
void matmul_cpu(int *a, int *b, int *c, int ra, int ca, int rb, int cb) {
    for (int i = 0; i < ra; i++) {
        for (int j = 0; j < cb; j++) {
            int sum = 0;
            for (int k = 0; k < ca; k++) {
                sum += a[i * ca + k] * b[k * cb + j];
            }
            c[i * cb + j] = sum;
        }
    }
}

// Initialize matrix with random values
void initMatrix(int *matrix, int rows, int cols, int seed) {
    srand(seed);
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = rand() % 10; // Values between 0-9
    }
}

// Print matrix (for small matrices only)
void printMatrix(int *matrix, int rows, int cols, const char* name) {
    std::cout << "\n" << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows && i < 8; i++) { // Limit to 8 rows for readability
        for (int j = 0; j < cols && j < 8; j++) { // Limit to 8 columns
            std::cout << matrix[i * cols + j] << "\t";
        }
        if (cols > 8) std::cout << "...";
        std::cout << "\n";
    }
    if (rows > 8) std::cout << "...\n";
}

// Verify results
bool verifyResults(int *c1, int *c2, int size) {
    for (int i = 0; i < size; i++) {
        if (c1[i] != c2[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    // Matrix dimensions - you can modify these
    const int ra = 512, ca = 512; // Matrix A: 512x512
    const int rb = 512, cb = 512; // Matrix B: 512x512
    
    // Verify dimensions are compatible
    if (ca != rb) {
        std::cerr << "Error: Incompatible matrix dimensions for multiplication" << std::endl;
        return -1;
    }
    
    std::cout << "Matrix Multiplication: A(" << ra << "x" << ca << ") × B(" << rb << "x" << cb << ")\n";
    std::cout << "Tile size: " << TILE_SIZE << "x" << TILE_SIZE << "\n\n";
    
    // Allocate host memory
    size_t size_a = ra * ca * sizeof(int);
    size_t size_b = rb * cb * sizeof(int);
    size_t size_c = ra * cb * sizeof(int);
    
    int *h_a = (int*)malloc(size_a);
    int *h_b = (int*)malloc(size_b);
    int *h_c_shared = (int*)malloc(size_c);
    int *h_c_naive = (int*)malloc(size_c);
    int *h_c_cpu = (int*)malloc(size_c);
    
    if (!h_a || !h_b || !h_c_shared || !h_c_naive || !h_c_cpu) {
        std::cerr << "Failed to allocate host memory" << std::endl;
        return -1;
    }
    
    // Initialize matrices
    std::cout << "Initializing matrices...\n";
    initMatrix(h_a, ra, ca, 42);
    initMatrix(h_b, rb, cb, 123);
    
    // Print small portion of input matrices
    if (ra <= 8 && ca <= 8) {
        printMatrix(h_a, ra, ca, "Matrix A");
        printMatrix(h_b, rb, cb, "Matrix B");
    }
    
    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    
    // Copy matrices to device
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    
    // Setup execution configuration
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((cb + TILE_SIZE - 1) / TILE_SIZE, (ra + TILE_SIZE - 1) / TILE_SIZE);
    
    std::cout << "Grid size: " << gridSize.x << "x" << gridSize.y << std::endl;
    std::cout << "Block size: " << blockSize.x << "x" << blockSize.y << "\n\n";
    
    // Warm up GPU
    matmul_shared<<<gridSize, blockSize>>>(d_a, d_b, d_c, ra, ca, rb, cb);
    cudaDeviceSynchronize();
    
    // Time shared memory version
    auto start = std::chrono::high_resolution_clock::now();
    
    matmul_shared<<<gridSize, blockSize>>>(d_a, d_b, d_c, ra, ca, rb, cb);
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_shared = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Copy result back
    cudaMemcpy(h_c_shared, d_c, size_c, cudaMemcpyDeviceToHost);
    
    std::cout << "✓ Shared memory tiled version completed in " << duration_shared.count() << " µs\n";
    
    // Time naive version for comparison
    start = std::chrono::high_resolution_clock::now();
    
    matmul_naive<<<gridSize, blockSize>>>(d_a, d_b, d_c, ra, ca, rb, cb);
    cudaDeviceSynchronize();
    
    end = std::chrono::high_resolution_clock::now();
    auto duration_naive = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    cudaMemcpy(h_c_naive, d_c, size_c, cudaMemcpyDeviceToHost);
    
    std::cout << "✓ Naive version completed in " << duration_naive.count() << " µs\n";
    
    // CPU version for small matrices (verification)
    if (ra <= 64 && cb <= 64) {
        std::cout << "Running CPU verification...\n";
        auto cpu_start = std::chrono::high_resolution_clock::now();
        
        matmul_cpu(h_a, h_b, h_c_cpu, ra, ca, rb, cb);
        
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
        
        std::cout << "✓ CPU version completed in " << cpu_duration.count() << " µs\n\n";
        
        // Verify results
        bool shared_correct = verifyResults(h_c_shared, h_c_cpu, ra * cb);
        bool naive_correct = verifyResults(h_c_naive, h_c_cpu, ra * cb);
        
        std::cout << "Verification Results:\n";
        std::cout << "Shared memory version: " << (shared_correct ? "✓ CORRECT" : "✗ INCORRECT") << "\n";
        std::cout << "Naive version: " << (naive_correct ? "✓ CORRECT" : "✗ INCORRECT") << "\n\n";
    } else {
        // For large matrices, just verify shared vs naive
        bool results_match = verifyResults(h_c_shared, h_c_naive, ra * cb);
        std::cout << "\nShared vs Naive comparison: " << (results_match ? "✓ MATCH" : "✗ DIFFER") << "\n\n";
    }
    
    // Performance comparison
    double speedup = (double)duration_naive.count() / duration_shared.count();
    std::cout << "Performance Summary:\n";
    std::cout << "Shared memory speedup: " << speedup << "x faster than naive\n";
    
    // Calculate throughput (operations per second)
    long long ops = (long long)ra * cb * ca * 2; // Each element requires ca multiplications and additions
    double gflops_shared = (double)ops / (duration_shared.count() * 1000.0); // GFLOPS
    double gflops_naive = (double)ops / (duration_naive.count() * 1000.0);
    
    std::cout << "Shared memory throughput: " << gflops_shared << " GFLOPS\n";
    std::cout << "Naive throughput: " << gflops_naive << " GFLOPS\n";
    
    // Print small portion of result
    if (ra <= 8 && cb <= 8) {
        printMatrix(h_c_shared, ra, cb, "Result Matrix C = A × B (Shared Memory)");
    }
    
    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c_shared);
    free(h_c_naive);
    free(h_c_cpu);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    std::cout << "\nMemory cleanup completed.\n";
    
    return 0;
}


/*
Matrix Multiplication: A(512x512) × B(512x512)
Tile size: 16x16

Initializing matrices...
Grid size: 32x32
Block size: 16x16

✓ Shared memory tiled version completed in 393 µs
✓ Naive version completed in 549 µs

Shared vs Naive comparison: ✓ MATCH

Performance Summary:
Shared memory speedup: 1.39695x faster than naive
Shared memory throughput: 683.042 GFLOPS
Naive throughput: 488.953 GFLOPS

Memory cleanup completed.
*/