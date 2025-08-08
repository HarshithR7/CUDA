/*


// ==================== VERSION 2: CONSTANT MEMORY OPTIMIZATION ====================
// Constant memory for small matrices (limited to ~64KB) 

__constant__ int const_mem[1024];

__global__ void matmul_const(int *a, int *b, int *c, int ra, int ca, int rb,int cb){

    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int j=blockIdx.y * blockDim.y + threadIdx.y;

    if (i<ra && j<cb){
        int sum=0;
        for (int k=0;k<ca;++k){
        // Fixed indexing: row-major for both matrices
            sum+=a[i*ca+k]*const_mem[k*cb+j]; // ca has no.of ekements in column of matrix a
        }
        c[i*cb+j]=sum;
    }
}
*/


#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>

// Constant memory declaration (limited to ~64KB = 16384 integers)
__constant__ int const_matrix_b[16384];

// Version 1: Constant memory optimization (Matrix B in constant memory)
__global__ void matmul_const(int *a, int *c, int ra, int ca, int rb, int cb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row
    int j = blockIdx.y * blockDim.y + threadIdx.y; // column

    if (i < ra && j < cb) {
        int sum = 0;
        for (int k = 0; k < ca; ++k) {
            // Matrix A from global memory, Matrix B from constant memory
            sum += a[i * ca + k] * const_matrix_b[k * cb + j];
        }
        c[i * cb + j] = sum;
    }
}

// Version 2: Matrix A in constant memory (alternative approach)
__global__ void matmul_const_alt(int *b, int *c, int ra, int ca, int rb, int cb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row
    int j = blockIdx.y * blockDim.y + threadIdx.y; // column

    if (i < ra && j < cb) {
        int sum = 0;
        for (int k = 0; k < ca; ++k) {
            // Matrix A from constant memory, Matrix B from global memory
            sum += const_matrix_b[i * ca + k] * b[k * cb + j];
        }
        c[i * cb + j] = sum;
    }
}

// Standard global memory version for comparison
__global__ void matmul_global(int *a, int *b, int *c, int ra, int ca, int rb, int cb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row
    int j = blockIdx.y * blockDim.y + threadIdx.y; // column

    if (i < ra && j < cb) {
        int sum = 0;
        for (int k = 0; k < ca; ++k) {
            sum += a[i * ca + k] * b[k * cb + j];
        }
        c[i * cb + j] = sum;
    }
}

// CPU reference implementation
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

// Initialize matrix with test values
void initMatrix(int *matrix, int rows, int cols, int seed) {
    srand(seed);
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (rand() % 20) - 10; // Values between -10 and 9
    }
}

// Print matrix (limited output for readability)
void printMatrix(int *matrix, int rows, int cols, const char* name) {
    std::cout << "\n" << name << " (" << rows << "x" << cols << "):\n";
    int max_print = std::min(8, std::min(rows, cols));
    
    for (int i = 0; i < max_print; i++) {
        for (int j = 0; j < max_print; j++) {
            std::cout << matrix[i * cols + j] << "\t";
        }
        if (cols > max_print) std::cout << "...";
        std::cout << "\n";
    }
    if (rows > max_print) std::cout << "...\n";
}

// Verify results between two matrices
bool verifyResults(int *result1, int *result2, int size, const char* name1, const char* name2) {
    for (int i = 0; i < size; i++) {
        if (result1[i] != result2[i]) {
            std::cout << "Mismatch at index " << i << ": " 
                      << name1 << "=" << result1[i] 
                      << ", " << name2 << "=" << result2[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Calculate maximum matrix size that fits in constant memory
void printConstantMemoryLimits() {
    std::cout << "=== Constant Memory Limitations ===\n";
    std::cout << "Constant memory size: ~64KB = 16384 integers\n";
    std::cout << "Maximum square matrix in constant memory: " 
              << (int)sqrt(16384) << "x" << (int)sqrt(16384) << std::endl;
    
    std::cout << "\nSuggested matrix sizes for constant memory:\n";
    std::cout << "32x32   = 1024 elements\n";
    std::cout << "64x64   = 4096 elements\n";
    std::cout << "128x128 = 16384 elements (maximum)\n\n";
}

int main() {
    printConstantMemoryLimits();
    
    // Matrix dimensions - keep small for constant memory
    // Matrix B will be stored in constant memory
    const int ra = 64, ca = 64;  // Matrix A: 64x64
    const int rb = 64, cb = 64;  // Matrix B: 64x64
    
    // Check if matrix B fits in constant memory
    if (rb * cb > 16384) {
        std::cerr << "Error: Matrix B (" << rb << "x" << cb << " = " << rb*cb 
                  << " elements) exceeds constant memory limit (16384 elements)" << std::endl;
        return -1;
    }
    
    // Verify dimensions are compatible
    if (ca != rb) {
        std::cerr << "Error: Incompatible matrix dimensions for multiplication" << std::endl;
        return -1;
    }
    
    std::cout << "Matrix Multiplication: A(" << ra << "x" << ca << ") × B(" << rb << "x" << cb << ")\n";
    std::cout << "Matrix B size: " << rb * cb << " elements (fits in constant memory)\n\n";
    
    // Calculate memory sizes
    size_t size_a = ra * ca * sizeof(int);
    size_t size_b = rb * cb * sizeof(int);
    size_t size_c = ra * cb * sizeof(int);
    
    // Allocate host memory
    int *h_a = (int*)malloc(size_a);
    int *h_b = (int*)malloc(size_b);
    int *h_c_const = (int*)malloc(size_c);
    int *h_c_global = (int*)malloc(size_c);
    int *h_c_cpu = (int*)malloc(size_c);
    
    if (!h_a || !h_b || !h_c_const || !h_c_global || !h_c_cpu) {
        std::cerr << "Failed to allocate host memory" << std::endl;
        return -1;
    }
    
    // Initialize matrices
    std::cout << "Initializing matrices...\n";
    initMatrix(h_a, ra, ca, 42);
    initMatrix(h_b, rb, cb, 123);
    
    // Print input matrices (if small enough)
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
    
    // Copy Matrix B to constant memory
    std::cout << "Copying Matrix B to constant memory...\n";
    cudaMemcpyToSymbol(const_matrix_b, h_b, size_b);
    
    // Setup execution configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((ra + blockSize.x - 1) / blockSize.x, 
                  (cb + blockSize.y - 1) / blockSize.y);
    
    std::cout << "Grid size: " << gridSize.x << "x" << gridSize.y << std::endl;
    std::cout << "Block size: " << blockSize.x << "x" << blockSize.y << "\n\n";
    
    // Warm up GPU
    matmul_const<<<gridSize, blockSize>>>(d_a, d_c, ra, ca, rb, cb);
    cudaDeviceSynchronize();
    
    std::cout << "=== Performance Testing ===\n";
    
    // Test 1: Constant Memory Version (Matrix B in constant memory)
    auto start = std::chrono::high_resolution_clock::now();
    
    matmul_const<<<gridSize, blockSize>>>(d_a, d_c, ra, ca, rb, cb);
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_const = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    cudaMemcpy(h_c_const, d_c, size_c, cudaMemcpyDeviceToHost);
    std::cout << "✓ Constant memory version: " << duration_const.count() << " µs\n";
    
    // Test 2: Global Memory Version
    start = std::chrono::high_resolution_clock::now();
    
    matmul_global<<<gridSize, blockSize>>>(d_a, d_b, d_c, ra, ca, rb, cb);
    cudaDeviceSynchronize();
    
    end = std::chrono::high_resolution_clock::now();
    auto duration_global = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    cudaMemcpy(h_c_global, d_c, size_c, cudaMemcpyDeviceToHost);
    std::cout << "✓ Global memory version: " << duration_global.count() << " µs\n";
    
    // Test 3: CPU Version (for verification)
    std::cout << "Running CPU verification...\n";
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    matmul_cpu(h_a, h_b, h_c_cpu, ra, ca, rb, cb);
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    
    std::cout << "✓ CPU version: " << cpu_duration.count() << " µs\n\n";
    
    // Verify results
    std::cout << "=== Verification ===\n";
    bool const_correct = verifyResults(h_c_const, h_c_cpu, ra * cb, "Constant", "CPU");
    bool global_correct = verifyResults(h_c_global, h_c_cpu, ra * cb, "Global", "CPU");
    
    std::cout << "Constant memory result: " << (const_correct ? "✓ CORRECT" : "✗ INCORRECT") << "\n";
    std::cout << "Global memory result: " << (global_correct ? "✓ CORRECT" : "✗ INCORRECT") << "\n\n";
    
    // Performance analysis
    std::cout << "=== Performance Analysis ===\n";
    double speedup_const = (double)duration_global.count() / duration_const.count();
    double speedup_gpu = (double)cpu_duration.count() / duration_const.count();
    
    std::cout << "Constant memory speedup vs Global memory: " << speedup_const << "x\n";
    std::cout << "Constant memory speedup vs CPU: " << speedup_gpu << "x\n";
    
    // Calculate throughput
    long long ops = (long long)ra * cb * ca * 2; // Each element: ca multiplications + ca additions
    double gflops_const = (double)ops / (duration_const.count() * 1000.0);
    double gflops_global = (double)ops / (duration_global.count() * 1000.0);
    double gflops_cpu = (double)ops / (cpu_duration.count() * 1000.0);
    
    std::cout << "\nThroughput Comparison:\n";
    std::cout << "Constant memory: " << gflops_const << " GFLOPS\n";
    std::cout << "Global memory: " << gflops_global << " GFLOPS\n";
    std::cout << "CPU: " << gflops_cpu << " GFLOPS\n";
    
    // Memory bandwidth analysis
    long long bytes_const = (long long)ra * ca * sizeof(int) + ra * cb * sizeof(int); // A from global + C to global
    long long bytes_global = (long long)(ra * ca + rb * cb + ra * cb) * sizeof(int); // A + B + C
    
    double bandwidth_const = (double)bytes_const / (duration_const.count());  // GB/s
    double bandwidth_global = (double)bytes_global / (duration_global.count()); // GB/s
    
    std::cout << "\nEffective Memory Bandwidth:\n";
    std::cout << "Constant memory version: " << bandwidth_const << " GB/s\n";
    std::cout << "Global memory version: " << bandwidth_global << " GB/s\n";
    
    // Print sample results
    if (ra <= 8 && cb <= 8) {
        printMatrix(h_c_const, ra, cb, "Result Matrix C = A × B (Constant Memory)");
    }
    
    // Summary
    std::cout << "\n=== Summary ===\n";
    std::cout << "Constant memory is effective when:\n";
    std::cout << "1. Matrix fits in ~64KB limit\n";
    std::cout << "2. Matrix is accessed multiple times by many threads\n";
    std::cout << "3. Access pattern has good spatial locality\n\n";
    
    if (speedup_const > 1.0) {
        std::cout << "✓ Constant memory provided " << speedup_const << "x speedup!\n";
    } else {
        std::cout << "ℹ Constant memory didn't show significant speedup for this case.\n";
        std::cout << "  This might be due to cache effects or matrix size.\n";
    }
    
    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c_const);
    free(h_c_global);
    free(h_c_cpu);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    std::cout << "\nMemory cleanup completed.\n";
    
    return 0;
}

/*
=== Constant Memory Limitations ===
Constant memory size: ~64KB = 16384 integers
Maximum square matrix in constant memory: 128x128

Suggested matrix sizes for constant memory:
32x32   = 1024 elements
64x64   = 4096 elements
128x128 = 16384 elements (maximum)

Matrix Multiplication: A(64x64) × B(64x64)
Matrix B size: 4096 elements (fits in constant memory)

Initializing matrices...
Copying Matrix B to constant memory...
Grid size: 4x4
Block size: 16x16

=== Performance Testing ===
✓ Constant memory version: 74 µs
✓ Global memory version: 112 µs
Running CPU verification...
✓ CPU version: 952 µs

=== Verification ===
Constant memory result: ✓ CORRECT
Global memory result: ✓ CORRECT

=== Performance Analysis ===
Constant memory speedup vs Global memory: 1.51351x
Constant memory speedup vs CPU: 12.8649x

Throughput Comparison:
Constant memory: 7.08497 GFLOPS
Global memory: 4.68114 GFLOPS
CPU: 0.550723 GFLOPS

Effective Memory Bandwidth:
Constant memory version: 442.811 GB/s
Global memory version: 438.857 GB/s

=== Summary ===
Constant memory is effective when:
1. Matrix fits in ~64KB limit
2. Matrix is accessed multiple times by many threads
3. Access pattern has good spatial locality

✓ Constant memory provided 1.51351x speedup!

Memory cleanup completed.
*/