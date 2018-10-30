#pragma once
#include "OMP_Multi.h"

void Multi(double* A, double* B, double* C, int numElements)
{
	#pragma omp parallel shared(A,B,C)
	{
		#pragma omp for
			for (int i = 0; i < numElements; i++)
			{
				for (int k = 0; k < numElements; k++)
				{
					for (int j = 0; j < numElements; j++)
						C[i*numElements + j] += A[i * numElements + k] * B[j + k * numElements];
				}
			}
		}
}