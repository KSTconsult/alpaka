#include <omp.h>
#include <iostream>
#include <random>
#include <chrono>
#include <math.h>

using namespace std;

void Multi_Block(int offi, int offj, int offk, int n, double* _A, double* _B, double* _C, int la, int lb);

void Free(double* matr, int n)
{
	delete[] matr;
}

double* MatrMalloc(int n)
{
	double* matr = new double[n * n];
	return matr;
}

void InitMatr(double* _A, double* _B, double* _C, int n)
{
	// Генератор случайных чисел
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(-2.00, 2.00);

	// Инициализация матриц
	for (int i = 0; i < n * n; i++)
	{
		double dice_roll = distribution(generator);
		_A[i] = dice_roll;
		dice_roll = distribution(generator);
		_B[i] = dice_roll;
		_C[i] = 0;
	}
}

void CopyMatr(double* a, double* b, int n)
{
	for (int i = 0; i < n * n; i++)
		b[i] = a[i];
}

void PrintMatr(double* matr, int n)
{
	for (int i = 0; i < n * n; i++)
	{
		cout << matr[i] << " ";
		if (i % n == 0 && i != 0)
			cout << endl;
	}
	cout << endl;
}

bool Check_Equal_Matr(double* a, double* b, int n)
{
	bool flag = true;

	for (int i = 0; i < n * n; i++)
	{
		if (fabs(a[i] - b[i]) > 1e-6)
		{
			flag = false;
			break;
		}
	}
	return flag;
}

void Block_Matr_Multi_OMP_v1(double* _A, double* _B, double* _C, int la, int lb, int n)
{
	int ii, kk, i, j, k, jj;
#pragma omp parallel shared(_A, _B, _C, n) private(ii,kk, jj, i, j, k)
	{
#pragma omp for
		// За итерацию ii цикла получаем строку рез. матрицы толщиной в размер блока
		for (ii = 0; ii < n / lb; ii++)
			for (int jj = 0; jj < n / la; jj++)
				for (int i = ii * lb; i < ii * lb + lb; i++)
					for (int k = jj * la; k < jj * la + la; k++)
						for (int j = 0; j < n; j++)
							_C[i * n + j] += _A[i * n + k] * _B[k * n + j];
	}
}

void Block_Matr_Multi_OMP_v2(double* _A, double* _B, double* _C, int la, int lb, int n)
{
	int ib, jb, kb;
	#pragma omp parallel shared(_A,_B,_C) private(ib,jb,kb)
	{
		#pragma omp for
		// Умножение блоков
		for (ib = 0; ib < n / lb; ib++)
		{
			cout << omp_get_num_threads() << endl;
			for (kb = 0; kb < n / la; kb++)
			{
				for (jb = 0; jb < n / la; jb++)
					// Умножение блоков
					Multi_Block(ib, jb, kb, n, _A, _B, _C, la, lb);
			}
		}
	}
}

void Multi_Block(int offi, int offj, int offk, int n, double* _A, double* _B, double* _C, int la, int lb)
{
	// Умножение блоков
	for (int i = offi * lb; i < offi*lb + lb; i++)
	{
		for (int k = offk * la; k < offk*la + la; k++)
		{
			const int start = offj * la;
			const int finish = start + la;

			const double AA = _A[i * n + k];
			double* const pc = &_C[i * n];
			const double* const pb = &_B[k * n];

			#pragma loop( ivdep ) 
			for (int j = start; j < finish; j++)
				pc[j] += AA * pb[j];
		}
	}
}

void Matr_Mulit_OMP(double* _A, double* _B, double* _C, int n)
{
	int i, j, k;
#pragma omp parallel shared(_A,_B,_C, n) private(i,j,k)
	{
#pragma omp for
		for (i = 0; i < n; i++)
		{
			for (k = 0; k < n; k++)
			{
				for (j = 0; j < n; j++)
					_C[i * n + j] += (_A[i * n + k] * _B[k * n + j]);
			}
		}
	}
}
