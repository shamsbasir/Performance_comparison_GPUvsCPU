#include "definitions.h"
#include "myFunctions.h"
#include "params.h"
#include "typedefs.h"




INT main(INT argc, char *argv[])
{
    if (argc < 4) {
        perror("Command-line usage: executableName <m><n><k>");
        exit(1);
    }

    INT m = atoi(argv[ 1 ]);
    INT n = atoi(argv[ 2 ]);
    INT k = atoi(argv[ 3 ]);

    REAL *a, *b, *c, *d;
    // Allocated Unified Memory -- accessible by CPU and GPU

    cudaMallocManaged(( void ** ) &a, sizeof(*a) * m * n);

    cudaMallocManaged(( void ** ) &b, sizeof(*b) * n * k);

    cudaMallocManaged(( void ** ) &c, sizeof(*c) * m * k);

    cudaMallocManaged(( void ** ) &d, sizeof(*d) * m * k);

    // initialize the matrix

    InitializeMatrices(a, b, m, n, k);
    float elapsedTime_gpu;
    cudaEvent_t timeStart, timeStop; // WARNING!!! use events only to time the device
    cudaEventCreate(&timeStart);
    cudaEventCreate(&timeStop);
    printf(GREEN"\n=====MATRIX A[%d X %d]==========\n"RESET, m, n);
    printMatrix(a, m, n);

    printf(GREEN"=====MATRIX B[%d X %d]=======\n"RESET, n, k);

    printMatrix(b, n, k);

    cudaEventRecord(timeStart, 0);

    dim3 dimBlock(XBLOCK, YBLOCK);

    dim3 dimGrid((k + XBLOCK - 1) / dimBlock.x, (m + YBLOCK - 1) / dimBlock.y);

    matrixMultiplyGPU_gl<<< dimGrid, dimBlock >>>(a, b, c, m, n, k);

    cudaEventRecord(timeStop, 0);
    cudaEventSynchronize(timeStop);

    cudaEventElapsedTime(&elapsedTime_gpu, timeStart, timeStop);
    printf(CYAN"***************** GPU KERNEL FUNCTION RESULT *******************\n"RESET);

    printMatrix(c, m, k);

    printf(GREEN"elapsed wall time (GPU) = "YELLOW"%5.5f  seconds"RESET"\n"RESET, elapsedTime_gpu/1000.f);


    cudaEventRecord(timeStart, 0);
    // cublas ddot() function is used for the inner product of rows and columns
    matrixMultipy_ddot(a, b, c, m, n, k);
    cudaEventRecord(timeStop, 0);
    cudaDeviceSynchronize( );
    cudaEventElapsedTime(&elapsedTime_gpu, timeStart, timeStop);
    printf(CYAN"***************** GPU DDOT FUNCTION RESULT *******************\n"RESET);
    printMatrix(c, m, k);
    printf(GREEN"elapsed wall time (GPU DDOT) = "YELLOW"%5.5f  seconds"RESET"\n"RESET, elapsedTime_gpu/1000.f);
   
    cudaEventRecord(timeStart, 0);
   // cublas daxpy() function is used to calculate columns of the resultant matrix
   // as a linear combination of the columns of the first matrix
    matrixMultipy_daxpy(a, b, d, m, n, k);
    cudaEventRecord(timeStop, 0);
    cudaDeviceSynchronize( );
    cudaEventElapsedTime(&elapsedTime_gpu, timeStart, timeStop);
    printf(CYAN"******************* GPU DAXPY FUNCTION RESULT *******************\n"RESET);
    printMatrix(d, m, k);
    printf(GREEN"elapsed wall time (GPU DAXPY) = "YELLOW"%5.5f  seconds"RESET"\n"RESET, elapsedTime_gpu/1000.f); 
   
 
    cudaEventDestroy(timeStart);
    cudaEventDestroy(timeStop);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(d);

    return EXIT_SUCCESS;
}
