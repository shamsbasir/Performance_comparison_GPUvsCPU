#ifndef _MYFUNCTIONS_H_
#define _MYFUNCTIONS_H_

#include "typedefs.h"


void printMatrix (REAL *c, INT nrow, INT ncol);

void InitializeMatrices(REAL *a, REAL *b, INT m, INT n, INT k);

void matrixMultiply(REAL *a,  REAL *b, REAL *c, INT m,INT n, INT k);


void ddot_Matrix_Mult(REAL *a, REAL *b,  REAL *c, INT m, INT n, INT k);

void daxpy_Matrix_Mult(REAL *a,REAL *b, REAL *c, INT m,INT n, INT k);

void dgemm_Matrix_Mult(REAL *a, REAL *b, REAL *c, INT m, INT n, INT k);

#endif
