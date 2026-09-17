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
	F U
)
{
	__shared__ NUMERIC_TYPE shared_min[CUDA_BLOCK_SIZE];
	const int tid = threadIdx.x;
	const int i = blockIdx.x * blockDim.x + tid;
	NUMERIC_TYPE local_min = cuda::solver_params.max_dt;

	if (i < cuda::geometry.xsz)
	{
		for (int j = blockIdx.y; j < cuda::geometry.ysz; j += gridDim.y)
		{
			const int k = (j + 1) * cuda::pitch + (i + 1);
			const NUMERIC_TYPE H = U.H[k];
			if (H > cuda::solver_params.DepthThresh)
			{
				const NUMERIC_TYPE inv_H = C(1.0) / H;
				const NUMERIC_TYPE u = U.HU[k] * inv_H;
				const NUMERIC_TYPE v = U.HV[k] * inv_H;
				const NUMERIC_TYPE wave = SQRT(cuda::physical_params.g * H);
				const NUMERIC_TYPE dt_x = cuda::solver_params.cfl * cuda::geometry.dx /
					(FABS(u) + wave);
				const NUMERIC_TYPE dt_y = cuda::solver_params.cfl * cuda::geometry.dy /
					(FABS(v) + wave);
				local_min = FMIN(local_min, FMIN(dt_x, dt_y));
			}
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
		block_min[blockIdx.y * gridDim.x + blockIdx.x] = shared_min[0];
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
xsz(geometry.xsz),
ysz(geometry.ysz),
reduction_grid(1, 1, 1),
reduction_elements(0)
{
	if (adaptive)
	{
		d_temp = nullptr;
		NUMERIC_TYPE* dummy_in = nullptr;
		NUMERIC_TYPE* dummy_out = nullptr;

		const int x_blocks = std::max(1, (xsz + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
		const int target_blocks = 4096;
		const int y_blocks = std::max(1, std::min(ysz,
				std::max(1, target_blocks / x_blocks)));
		reduction_grid = dim3(x_blocks, y_blocks, 1);
		reduction_elements = x_blocks * y_blocks;

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
void lis::cuda::DynamicTimestep<F>::update_dt_async(cudaStream_t stream)
{
	if (adaptive)
	{
		auto& U = solver.d_U();
		lis::cuda::update_dt_block_min<F><<<reduction_grid, CUDA_BLOCK_SIZE, 0, stream>>>(
				dt_field, U);
		checkCudaErrors(cudaPeekAtLastError());
		checkCudaErrors(cub::DeviceReduce::Min(d_temp, bytes,
					dt_field, &dt, reduction_elements, stream));
	}
}

template<typename F>
NUMERIC_TYPE lis::cuda::DynamicTimestep<F>::update_dt()
{
	update_dt_async(0);
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