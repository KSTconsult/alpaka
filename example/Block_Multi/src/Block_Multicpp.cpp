#include <omp.h>
#include <iostream>
#include <random>
#include <chrono>

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

int main(int argc, char *argv[])
{
	int N = 1024; // Размер матрицы
	int La = 64; // Размер блока по длине
	int Lb = 64; // Размер блока по ширине
	int num_threads = 4;
	//int Na = N / La // Кол-во блоков в матрице по длине
	//int Nb = N / Lb; // Кол-во блоков в матрице по ширине

	if (argc > 1)
	{
		N = atoi(argv[1]);
		La = atoi(argv[2]);
		Lb = atoi(argv[3]);
		num_threads = atoi(argv[4]);
	}

	if ((N % La != 0) || (N % Lb != 0))
		exit(0);

	 // Время
	std::chrono::time_point<std::chrono::system_clock> start, end;

	double *A;
	double *B;
	double *C;
	double* C_true;

	A = MatrMalloc(N);
	B = MatrMalloc(N);
	C = MatrMalloc(N);
	C_true = MatrMalloc(N);

	InitMatr(A, B, C, N);

	//cout << "Inichializtion finished" << endl;

	CopyMatr(C, C_true, N);

	//PrintMatr(A, N);
	//PrintMatr(B, N);

	cout << omp_get_max_threads() << endl;
	omp_set_num_threads(num_threads);

	start = std::chrono::system_clock::now();
	//Block_Matr_Multi_OMP_v1(A, B, C, La, Lb, N);
	Block_Matr_Multi_OMP_v2(A, B, C, La, Lb, N);
	end = std::chrono::system_clock::now();

	cout << N << " " << La << " " << Lb << endl;

	cout << "Work time " << std::chrono::duration_cast<std::chrono::milliseconds>
		(end - start).count() << " millisec" << endl;

	//PrintMatr(C, N);

	Matr_Mulit_OMP(A, B, C_true, N);
	//PrintMatr(C_true, N);
	
	if (Check_Equal_Matr(C, C_true, N))
		cout << "all right" << endl;
	else
		cout << "remark" << endl;

	Free(A);
	Free(B);
	Free(C);

	//int n;
	//cin >> n;


	return 0;
}