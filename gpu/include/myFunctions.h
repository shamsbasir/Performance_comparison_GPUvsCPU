#ifndef _MYFUNCTIONS_H_
#define _MYFUNCTIONS_H_

#include "typedefs.h"

void InitializeMatrices(REAL *a, REAL *b, INT m, INT n, INT k);

void printMatrix(REAL *c, INT  nrow, INT  ncol);

void matrixMultiplyCPU(REAL *a, REAL *b, REAL *c, INT  m, INT  n, INT  k);

__global__ void matrixMultiplyGPU_gl(REAL *a, REAL *b, REAL *c, INT  m, INT  n, INT  k);

void matrixMultipy_ddot(REAL *a, REAL *b, REAL *c, INT  m, INT  n, INT  k);

void matrixMultipy_daxpy(REAL *a, REAL *b, REAL *c, INT  m, INT  n, INT  k);


#endif 
