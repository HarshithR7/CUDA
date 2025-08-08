#include <cstdio>


__global__ void matmul(int *a, int *b, int *c, int ra, int ca, int rb,int cb){

    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int j=blockIdx.y * blockDim.y + threadIdx.y;

    if (i<ra && j<cb){
        int sum=0;
        for (int k=0;k<ca;++k){
        // Fixed indexing: row-major for both matrices
            sum+=a[i*ca+k]*b[k*cb+j]; // ca has no.of ekements in column of matrix a
        }
        c[i*cb+j]=sum;
    }
}


__global__ void initialize_data(int *a, int *b, int ra, int ca, int rb,int cb){

    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int j=blockIdx.y * blockDim.y + threadIdx.y;
    if (i<ra && j<ca) {
        int idx=i*ca+j;
        a[idx]=idx;
        
    }
    if (i<rb && j<cb) {
        int idx=i*cb+j;
        b[idx]=2*idx;
    }
}


int main(){
    int ra,rb,ca,cb,rc,cc; 
    ra=rb=ca=cb=rc=cc=5;

    //host memory
    int *a= (int*)malloc(ra*ca*sizeof(int));
    int *b= (int*)malloc(rb*cb*sizeof(int));
    int *c= (int*)malloc(rc*cc*sizeof(int));
    int *da,*db,*dc;

    //device memory
    cudaMalloc((void**)&da,ra*ca*sizeof(int));
    cudaMalloc((void**)&db,rb*cb*sizeof(int));
    cudaMalloc((void**)&dc,rc*cc*sizeof(int));


    // copy from host to device

    cudaMemcpy(da,a,ra*ca*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(db,b,rb*cb*sizeof(int),cudaMemcpyHostToDevice);
    

    dim3 threads(16,16);
    dim3 blocks((max(ca,cb)+threads.x-1)/threads.x,(max(ra,rb)+threads.y-1)/threads.y );
    initialize_data<<<blocks,threads>>>(da,db,ra,ca,rb,cb);
    cudaDeviceSynchronize();

    matmul<<<blocks,threads>>>(da,db,dc,ra,ca,rb,cb);
    cudaDeviceSynchronize();

    cudaMemcpy(c,dc,rc*cc*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(b, db, rb * cb * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(a, da, ra * ca * sizeof(int), cudaMemcpyDeviceToHost);

    printf("=== Matrix A ===\n");
    for (int i = 0; i < ra; ++i) {
        for (int j = 0; j < ca; ++j) {
            printf("%4d ", a[i * ca + j]);
        }
        printf("\n");
    }
    
    printf("\n=== Matrix B ===\n");
    for (int i = 0; i < rb; ++i) {
        for (int j = 0; j < cb; ++j) {
            printf("%4d ", b[i * cb + j]);
        }
        printf("\n");
    }
    
    printf("\n=== Result Matrix C ===\n");
    for (int i = 0; i < rc; ++i) {
        for (int j = 0; j < cc; ++j) {
            printf("%4d ", c[i * cc + j]);
        }
        printf("\n");
    }
    


    free(a);
    free(b);
    free(c);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    return 0;

}


/* 3x3
=== Matrix A ===
   0    1    2
   3    4    5
   6    7    8

=== Matrix B ===
   0    2    4
   6    8   10
  12   14   16

=== Result Matrix C ===
  30   36   42
  84  108  132
 138  180  222
*/

/* 5x5
=== Matrix A ===
   0    1    2    3    4
   5    6    7    8    9
  10   11   12   13   14
  15   16   17   18   19
  20   21   22   23   24

=== Matrix B ===
   0    2    4    6    8
  10   12   14   16   18
  20   22   24   26   28
  30   32   34   36   38
  40   42   44   46   48

=== Result Matrix C ===
 300  320  340  360  380
 800  870  940 1010 1080
1300 1420 1540 1660 1780
1800 1970 2140 2310 2480
2300 2520 2740 2960 3180
*/