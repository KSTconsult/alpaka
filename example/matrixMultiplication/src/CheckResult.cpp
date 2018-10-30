#pragma once
#include "CheckResult.h"

void Check(int numElements, double* memBufHostC, double* memBufHostC_OMP)
{
	bool flag = true;

	for (int i = 0; i < numElements * numElements; i++)
	{
		if (memBufHostC_OMP[i] != memBufHostC[i])
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