#include "ghostraster.h"
#include "cuda_geometry.cuh"
#include "cuda_util.cuh"
#include <helper_cuda.h>
#include <cub/cub.cuh>
#include <algorithm>

namespace lis
{
namespace cuda
{

template<typename F>
__global__ void update_dt_block_min
(
	NUMERIC_TYPE* block_min,
	F U,
	int elements
)
{
	__shared__ NUMERIC_TYPE shared_min[CUDA_BLOCK_SIZE];
	const int tid = threadIdx.x;
	const int first = blockIdx.x * blockDim.x + tid;
	const int stride = blockDim.x * gridDim.x;
	NUMERIC_TYPE local_min = cuda::solver_params.max_dt;

	for (int k = first; k < elements; k += stride)
	{
		const NUMERIC_TYPE H = U.H[k];
		if (H > cuda::solver_params.DepthThresh)
		{
			const NUMERIC_TYPE u = U.HU[k] / H;
			const NUMERIC_TYPE v = U.HV[k] / H;
			const NUMERIC_TYPE wave = SQRT(cuda::physical_params.g * H);
			const NUMERIC_TYPE dt_x = cuda::solver_params.cfl * cuda::geometry.dx /
				(FABS(u) + wave);
			const NUMERIC_TYPE dt_y = cuda::solver_params.cfl * cuda::geometry.dy /
				(FABS(v) + wave);
			local_min = FMIN(local_min, FMIN(dt_x, dt_y));
		}
	}

	shared_min[tid] = local_min;
	__syncthreads();
	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if (tid < offset)
			shared_min[tid] = FMIN(shared_min[tid], shared_min[tid + offset]);
		__syncthreads();
	}
	if (tid == 0)
		block_min[blockIdx.x] = shared_min[0];
}

}
}

template<typename F>
lis::cuda::DynamicTimestep<F>::DynamicTimestep
(
	NUMERIC_TYPE& dt,
	Geometry& geometry,
	NUMERIC_TYPE max_dt,
	int adaptive_ts,
	Solver<F>& solver
)
:
dt(dt),
max_dt(max_dt),
adaptive(adaptive_ts == ON),
solver(solver),
elements(lis::GhostRaster::elements(geometry)),
reduction_elements(0)
{
	if (adaptive)
	{
		d_temp = nullptr;
		NUMERIC_TYPE* dummy_in = nullptr;
		NUMERIC_TYPE* dummy_out = nullptr;

		const int natural_blocks = (elements + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
		reduction_elements = std::max(1, std::min(natural_blocks, 4096));

		checkCudaErrors(cub::DeviceReduce::Min(d_temp, bytes,
					dummy_in, dummy_out, reduction_elements));

		d_temp = malloc_device(bytes);
		dt_field = static_cast<NUMERIC_TYPE*>(
				malloc_device(reduction_elements * sizeof(NUMERIC_TYPE)));
	}
	else
	{
		dt = max_dt;
	}
}

template<typename F>
lis::cuda::DynamicTimestepACC<F>::DynamicTimestepACC
(
	NUMERIC_TYPE& dt,
	Geometry& geometry,
	NUMERIC_TYPE max_dt,
	int adaptive_ts,
	Solver<F>& solver
)
	:
	dt(dt),
max_dt(max_dt),
adaptive(adaptive_ts == ON),
solver(solver),
elements(lis::GhostRaster::elements_H(geometry))
{

		d_temp = nullptr;
		NUMERIC_TYPE* dummy_in = nullptr;
		NUMERIC_TYPE* dummy_out = nullptr;

		checkCudaErrors(cub::DeviceReduce::Min(d_temp, bytes,
			dummy_in, dummy_out, elements));

		d_temp = malloc_device(bytes);

		dt_field = cuda::GhostRaster::allocate_device_H(geometry);


}

template<typename F>
NUMERIC_TYPE lis::cuda::DynamicTimestep<F>::update_dt()
{
	if (adaptive)
	{
		auto& U = solver.d_U();
		lis::cuda::update_dt_block_min<F><<<reduction_elements, CUDA_BLOCK_SIZE>>>(
				dt_field, U, elements);
		checkCudaErrors(cudaPeekAtLastError());
		checkCudaErrors(cub::DeviceReduce::Min(d_temp, bytes,
					dt_field, &dt, reduction_elements));
	}
	cuda::sync();

	return dt;
}

template<typename F>
NUMERIC_TYPE lis::cuda::DynamicTimestepACC<F>::update_dt_ACC()
{
		solver.update_dt_per_element(dt_field);
		checkCudaErrors(cub::DeviceReduce::Min(d_temp, bytes,
					dt_field, &dt, elements));

	cuda::sync(); 

	return dt;
}

template<typename F>
lis::cuda::DynamicTimestep<F>::~DynamicTimestep()
{
	if (adaptive)
	{
		free_device(dt_field);
		free_device(d_temp);
	}
}

template<typename F>
lis::cuda::DynamicTimestepACC<F>::~DynamicTimestepACC()
{

		cuda::free_device(dt_field);
		free_device(d_temp);

}