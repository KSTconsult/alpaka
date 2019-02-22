#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <alpaka/alpaka.hpp>

using namespace std;

void Check(int numElements, double* memBufHostC, double* memBufHostC_OMP);
void Multi(double* A, double* B, double* C, int numElements);

class Kernel // Kernel, which will execute each thread
{
public:
	//-----------------------------------------------------------------------------
	ALPAKA_NO_HOST_ACC_WARNING
		template<
		typename TAcc,
		typename TElem,
		typename TSize>
		ALPAKA_FN_ACC auto operator()(
			TAcc const & acc,
			TElem *A,
			TElem *B,
			TElem *C,
			TSize &N,
			TSize &la,
			TSize &lb) const
		-> void
	{
		// Define the tread`s index
		auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);

		// Elemnts per thread (How many rows multi one thread)
		auto const threadElemExtentRow(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);

		unsigned int numRows = N;
		unsigned int numCols = N;

		// begin&end row for thread
		auto beginRow(gridThreadIdx * threadElemExtentRow);
		if (beginRow > numRows)
			beginRow = numRows;
		auto endRow(beginRow + threadElemExtentRow);
		if (endRow > numRows)
			endRow = numRows;

		//cout << "Index" << gridThreadIdx << " " << beginRow << " " << endRow << endl;// << gridThreadIdy << " "  " " << beginСol << endCols << endl;

		for (unsigned int ib = beginRow; (ib < endRow) && (ib < N / lb); ib++)
			for (unsigned int kb = 0; kb < N / la; kb++)
				for (unsigned int jb = 0; jb < N / la; jb++)
				{ // Multi Blocks
					for (unsigned int i = ib * lb; i < ib*lb + lb; i++)
						for (unsigned int k = kb * la; k < kb*la + la; k++)
						{
							const int start = jb * la;
							const int finish = start + la;

							const double AA = A[i * N + k];
							double* const pc = &C[i * N];
							const double* const pb = &B[k * N];

							#pragma ivdep
							for (int j = start; j < finish; j++)
								pc[j] += AA * pb[j];
						}
				}
	}

};

//using WorkDiv2 = alpaka::workdiv::WorkDivMembers<Dim, Size>;

auto main(int argc, char *argv[])
-> int
{


	using Val = double;
	using Size = std::size_t;
	//using Size = std::size_t

	int N = 2048;
	int LA = 64;
	int LB = 64;
	int num_threads = 4;
	if (argc > 1)
	{
		N = atoi(argv[1]);
		LA = atoi(argv[2]);
		LB = atoi(argv[3]);
		num_threads = atoi(argv[4]);
	}
	
	cout << omp_get_max_threads() << endl;
	omp_set_num_threads(num_threads);
	cout << omp_get_num_threads() << endl;

	// Size of matrix
	Size const numElements(N);
	Size const la(LA); // По сути не нужно
	Size const lb(LB);

	using Dim = alpaka::dim::DimInt<1>;
	using WorkDiv1 = alpaka::workdiv::WorkDivMembers<Dim, Size>;
	
	int _elemetsPerThread;
	//int threadsPerBlock;
	int _blocksPerGrid;
	
	if((N / LB) % num_threads == 0)
		_elemetsPerThread = (N / LB) / num_threads;
	else
		_elemetsPerThread = ((N / LB) / num_threads) + 1;
	
	if ((N / LB) % _elemetsPerThread == 0)
		_blocksPerGrid = (N / LB) /  _elemetsPerThread;
	else
		_blocksPerGrid = ((N / LB) /  _elemetsPerThread) + 1;
	
	alpaka::vec::Vec<Dim, Size> const elementsPerThread(
		static_cast<Size>(_elemetsPerThread));

	alpaka::vec::Vec<Dim, Size> const threadsPerBlock(
		static_cast<Size>(1));

	alpaka::vec::Vec<Dim, Size> const blocksPerGrid(
		static_cast<Size>(_blocksPerGrid));
	// +1 если количество элементов не делится на цело на кол-во threads или блоков

	WorkDiv1 workdiv(
		blocksPerGrid,
		threadsPerBlock,
		elementsPerThread);
	
	
	
	using Acc = alpaka::acc::AccCpuOmp2Blocks<alpaka::dim::DimInt<1u>, Size>;
	/*------*/

	using DevAcc = alpaka::dev::Dev<Acc>;
	using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
	using QueueAcc = alpaka::queue::QueueCpuSync;
	using PltfHost = alpaka::pltf::PltfCpu;

	// Create the kernel function object.
	Kernel Kernel;

	// Get the host device.
	auto const devHost(
		alpaka::pltf::getDevByIdx<PltfHost>(0u));

	// Select a device to execute on.
	auto const devAcc(
		alpaka::pltf::getDevByIdx<PltfAcc>(0u));

	// Get a stream on this device.
	QueueAcc stream(devAcc);

	// The data extent.
	alpaka::vec::Vec<alpaka::dim::DimInt<1u>, Size> const extent(
		numElements);

	//using Dim = alpaka::dim::DimInt<2>;

	std::cout
		<< "VectorAddKernelTester("
		<< " numElements:" << numElements
		<< ", accelerator: " << alpaka::acc::getAccName<Acc>()
		<< ", kernel: " << typeid(Kernel).name()
		<< ", workDiv: " << workdiv
		<< ")" << std::endl;

	// Allocate host memory buffers.
	auto memBufHostA(alpaka::mem::buf::alloc<Val, Size>(devHost, extent*extent));
	auto memBufHostB(alpaka::mem::buf::alloc<Val, Size>(devHost, extent*extent));
	auto memBufHostC(alpaka::mem::buf::alloc<Val, Size>(devHost, extent*extent));

	// How do it another way?
	double* A = new double[numElements*numElements];
	double* B = new double[numElements*numElements];
	double* memBufHostC_OMP = new double[numElements*numElements];

	// Random number generator
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(0, 10.00);
	double dice_roll;

	// Initialize the host input vectors
	for (Size i(0); i < numElements*numElements; ++i)
	{
		dice_roll = distribution(generator);
		alpaka::mem::view::getPtrNative(memBufHostA)[i] = dice_roll;
		dice_roll = distribution(generator);
		alpaka::mem::view::getPtrNative(memBufHostB)[i] = dice_roll;
		alpaka::mem::view::getPtrNative(memBufHostC)[i] = static_cast<Val>(0);
		memBufHostC_OMP[i] = static_cast<Val>(0);
	}

	// Allocate the buffers on the accelerator.
	auto memBufAccA(alpaka::mem::buf::alloc<Val, Size>(devAcc, extent*extent));
	auto memBufAccB(alpaka::mem::buf::alloc<Val, Size>(devAcc, extent*extent));
	auto memBufAccC(alpaka::mem::buf::alloc<Val, Size>(devAcc, extent*extent));

	// Copy Host -> Acc.
	alpaka::mem::view::copy(stream, memBufAccA, memBufHostA, extent*extent);
	alpaka::mem::view::copy(stream, memBufAccB, memBufHostB, extent*extent);
	alpaka::mem::view::copy(stream, memBufAccC, memBufHostC, extent*extent);

	// Create the executor task.

	auto const exec(alpaka::kernel::createTaskExec<Acc>(
		workdiv,
		Kernel,
		alpaka::mem::view::getPtrNative(memBufAccA),
		alpaka::mem::view::getPtrNative(memBufAccB),
		alpaka::mem::view::getPtrNative(memBufAccC),
		numElements, la, lb));

	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	// Profile the kernel execution.
	alpaka::queue::enqueue(stream, exec);
	end = std::chrono::system_clock::now();

	cout << "Work time " << std::chrono::duration_cast<std::chrono::milliseconds>
		(end - start).count() << " millisec" << endl;


	// Copy back the result.
	alpaka::mem::view::copy(stream, memBufHostC, memBufAccC, extent*extent);

	auto const pHostData(alpaka::mem::view::getPtrNative(memBufHostC));
	auto const pHostDataA(alpaka::mem::view::getPtrNative(memBufHostA));
	auto const pHostDataB(alpaka::mem::view::getPtrNative(memBufHostB));

	// Prepare for checking result
	double* Check_C = new double[numElements*numElements];

	for (unsigned int i = 0; i < numElements*numElements; i++)
	{
		Check_C[i] = pHostData[i];
		A[i] = pHostDataA[i];
		B[i] = pHostDataB[i];
	}

	// Multi OpenMP
	Multi(A, B, memBufHostC_OMP, numElements);

	// Check result
	Check(numElements, Check_C, memBufHostC_OMP);

	//int z;
	//cin >> z;

	return EXIT_SUCCESS;
}