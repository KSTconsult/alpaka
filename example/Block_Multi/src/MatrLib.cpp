#include <omp.h>
#include <iostream>
#include <random>
#include <chrono>
#include <math.h>

using namespace std;

void Free(double* matr);
double* MatrMalloc(int n);
void InitMatr(double* _A, double* _B, double* _C, int n);
void CopyMatr(double* a, double* b, int n);
void PrintMatr(double* b, int n);
bool Check_Equal_Matr(double* a, double* b, int n);
void Matr_Mulit_OMP(double* _A, double* _B, double* _C, int n);
void Block_Matr_Multi_OMP_v2(double* _A, double* _B, double* _C, int la, int lb, int n);
inline void Multi_Block(int offi, int offj, int offk, int n, double* _A, double* _B, double* _C, int la, int lb);

void Free(double* matr)
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

 void Block_Matr_Multi_OMP_v2(double* _A, double* _B, double* _C, int la, int lb, int n)
{
	int ib, jb, kb;
	#pragma omp parallel shared(_A,_B,_C) private(ib,jb,kb)
	{
		//cout << omp_get_num_threads() << endl;
		#pragma omp for
		// Умножение блоков
		for (ib = 0; ib < n / lb; ib++)
		{
			for (kb = 0; kb < n / la; kb++)
			{
				for (jb = 0; jb < n / la; jb++)
					// Умножение блоков
					Multi_Block(ib, jb, kb, n, _A, _B, _C, la, lb);
			}
		}
	}
}

inline void Multi_Block(int offi, int offj, int offk, int n, double* _A, double* _B, double* _C, int la, int lb)
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

			#pragma ivdep 
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
				#pragma ivdep
				for (j = 0; j < n; j++)
					_C[i * n + j] += (_A[i * n + k] * _B[k * n + j]);
			}
		}
	}
}
