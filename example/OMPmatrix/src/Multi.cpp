#include <iostream>
#include <omp.h>
#include <random>
#include <chrono>


using namespace std;

int main(int argc, char *argv[])
{
	int N = 1024;
	int num_threads = 4;
	if (argc > 1)
	{
		N = atoi(argv[1]);
		num_threads = atoi(argv[2]);
	}
	double *A = new double[N*N];
	double *B = new double[N*N];
	double *C = new double[N*N];




	// Генератор случайных чисел
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(-2.00, 2.00);

	// Время
	std::chrono::time_point<std::chrono::system_clock> start, end;

	// Инициализация матриц
	for (int i = 0; i < N * N; i++)
	{
			double dice_roll = distribution(generator);
			A[i] = dice_roll;
			dice_roll = distribution(generator);
			B[i] = dice_roll;
			C[i] = 0;
	}

	cout << "Inichializtion finished" << endl;

	omp_set_num_threads(num_threads);
	int max = omp_get_num_threads();
	cout << max << endl;

	start = std::chrono::system_clock::now();
	#pragma omp parallel shared(A,B,C)
	{
		#pragma omp for
		for (int i = 0; i < N; i++)
		{
			for (int k = 0; k < N; k++)
			{
				const int const_i = i;
				const int const_k = k;

				const double AA = A[i * N + k];
				double* const pc = &C[i * N];
				const double* const pb = &B[k * N];
				
				#pragma ivdep
				for (int j = 0; j < N; j++)
					pc[j] += (AA * pb[j]);
			}
		}
	}
	end = std::chrono::system_clock::now();

	cout << "Work time " << std::chrono::duration_cast<std::chrono::milliseconds>
		(end - start).count() << " millisec" << endl;

	return 0;
}




