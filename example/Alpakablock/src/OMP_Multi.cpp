#pragma once
#include <omp.h>

void Multi(double* A, double* B, double* C, int numElements)
{
	int i, j, k;
	#pragma omp parallel shared(A,B,C) private(i,j,k)
	{
		#pragma omp for
			for (i = 0; i < numElements; i++)
			{
				for (k = 0; k < numElements; k++)
				{
					for (j = 0; j < numElements; j++)
						C[i*numElements + j] += A[i * numElements + k] * B[j + k * numElements];
				}
			}
		}
}