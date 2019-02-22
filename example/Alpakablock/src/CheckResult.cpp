#pragma once
#include <iostream>
#include <math.h>

void Check(int numElements, double* memBufHostC, double* memBufHostC_OMP)
{
	bool flag = true;

	for (int i = 0; i < numElements * numElements; i++)
	{
		if (fabs(memBufHostC_OMP[i] - memBufHostC[i]) > 1e-6)
		{
			flag = false;
			break;
		}
	}

	if (flag)
		std::cout << "Correct!" << std::endl;
	else
		std::cout << "Inncorect!" << std::endl;
}