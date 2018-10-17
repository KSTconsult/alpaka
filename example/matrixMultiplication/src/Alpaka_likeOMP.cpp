#include <chrono>
#include <cmath>
#include <alpaka/alpaka.hpp>

using namespace std;

void Multi(double* A, double* B, double* C, int numElements);
void Check(int numElements, double* memBufHostC, double* memBufHostC_OMP);

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
			TSize &N) const
		-> void
	{
		// Define the tread`s index
		auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);

		// Elemnts per thread (How many rows multi one thread)
		auto const threadElemExtentRow(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);

		unsigned int numRows = N;

		// begin&end row for thread
		auto beginRow(gridThreadIdx * threadElemExtentRow);
		if (beginRow > numRows)
			beginRow = numRows;
		auto endRow(beginRow + threadElemExtentRow);
		if (endRow > numRows)
			endRow = numRows;

		for (unsigned int i = beginRow; (i < endRow) && (i < numRows); i++)
		{
			for (unsigned int k = 0; k < N; k++)
			{
				for (unsigned int j = 0; j < N; j++)
				{
					C[i*numRows + j] += A[i * numRows + k] * B[k * numRows + j];
				}
			}
		}
	}



};

auto main(int argc, char *argv[])
-> int
{
	using Size = std::size_t;
	using Dim = alpaka::dim::DimInt<1>;
	using WorkDiv1 = alpaka::workdiv::WorkDivMembers<Dim, Size>;

	using Val = double;


	// Size of matrix
	int N = 1024;
	if (argc > 1)
	{
		N = atoi(argv[1]);
	}

	Size const numElements(N);

	/*Create WorkDiv*/
	alpaka::vec::Vec<Dim, Size> const elementsPerThread(
		static_cast<Size>(numElements / 4));

	alpaka::vec::Vec<Dim, Size> const threadsPerBlock(
		static_cast<Size>(1));

	alpaka::vec::Vec<Dim, Size> const blocksPerGrid(
		static_cast<Size>(numElements));

	WorkDiv1 workdiv(
		blocksPerGrid,
		threadsPerBlock,
		elementsPerThread);

	/*-------*/

	using Acc = alpaka::acc::AccCpuOmp2Blocks<alpaka::dim::DimInt<1u>, Size>;
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

	// For check result
	double* A = new double[numElements*numElements];
	double* B = new double[numElements*numElements];
	double* memBufHostC_OMP = new double[numElements*numElements];

	// Random number generator
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(-2.00, 2.00);
	double dice_roll;

	// Initialize the host input vectors
	for (Size i(0); i < numElements*numElements; ++i)
	{
		dice_roll = distribution(generator);
		alpaka::mem::view::getPtrNative(memBufHostA)[i] = static_cast<Val>(dice_roll);
		A[i] = dice_roll;
		dice_roll = distribution(generator);
		alpaka::mem::view::getPtrNative(memBufHostB)[i] = static_cast<Val>(dice_roll);
		B[i] = dice_roll;
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
		numElements));

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
	// Prepare for checking result
	double* Check_C = new double[numElements*numElements];

	for (unsigned int i = 0; i < numElements*numElements; i++)
	{
		Check_C[i] = pHostData[i];
	}

	// Multi OpenMP
	Multi(A, B, memBufHostC_OMP, numElements);

	// Check result
	Check(numElements, Check_C, memBufHostC_OMP);

	return EXIT_SUCCESS;
}