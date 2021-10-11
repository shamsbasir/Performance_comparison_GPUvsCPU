#include "definitions.h"
#include "myFunctions.h"
#include "params.h"
#include "typedefs.h"

void InitializeMatrices(REAL *a, REAL *b, INT m, INT n, INT k)
{
    INT i, j, l, idx;

    // initialize matrices a & b

    for (i = 0; i < m; i++) {
        for (l = 0; l < n; l++) {
            idx      = l + i * n;
            a[ idx ] = rand()%10+1;
        }
    }

    for (l = 0; l < n; l++) {
        for (j = 0; j < k; j++) {
            idx      = j + l * k;
            b[ idx ] = rand()%10+1;;
        }
    }
}

void printMatrix(REAL *c, int nrow, int ncol)
{
#if (OUTOFF)
    int i, j, idx;
    for (i = 0; i < nrow; i++) {
        for (j = 0; j < ncol; j++) {
            idx = j + i * ncol;
            printf("%8.2f ; ", c[ idx ]);
        }
        printf("\n");
    }
    printf("\n");
#endif
}

__global__ void matrixMultiplyGPU_gl(REAL *a, REAL *b, REAL *c, int m, int n, int k)

{
    // Block index

    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row index of matrices a and c

    int row = by * YBLOCK + ty;

    // Column index of matrices a and b
    int col = bx * XBLOCK + tx;

    REAL C_temp = 0.0;

    if (row < m && col < k) {
        for (int l = 0; l < n; l++)
            C_temp += a[ l + row * n ] * b[ col + l * k ];

        c[ col + row * k ] = C_temp;
    }
}

void matrixMultipy_ddot(REAL *a, REAL *b, REAL *c, int m, int n, int k)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    REAL result;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // cublas ddot function that provides the result of
            // the dot product of two arrays (e.g. a row of A and
            // a column of B)
            cublasDdot(handle, n, a + i * n, 1, b + j, k, &result);

            c[ j + i * k ] = result;
        }
    }
    cublasDestroy(handle);
}

void matrixMultipy_daxpy(REAL *a, REAL *b, REAL *c, int m, int n, int k)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            REAL bb = b[ i + j * k ];
            // similar to cblas daxpy this function provides the columns of C
            // as a linear combinations of columns of A.
            // since a,b,c are inputted as 1D arrays of row-major,
            // a and c have n and k strides respectively.
            cublasDaxpy(handle, m, &bb, a + j, n, c + i, k);
        }
    }
    cublasDestroy(handle);
}
