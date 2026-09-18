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

template<typename F>
__global__ void update_dt_block_min_diag
(
	NUMERIC_TYPE* block_min,
	int* block_index,
	F U
)
{
	__shared__ NUMERIC_TYPE shared_min[CUDA_BLOCK_SIZE];
	__shared__ int shared_index[CUDA_BLOCK_SIZE];
	const int tid = threadIdx.x;
	const int i = blockIdx.x * blockDim.x + tid;
	NUMERIC_TYPE local_min = cuda::solver_params.max_dt;
	int local_index = -1;

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
				const NUMERIC_TYPE candidate = FMIN(dt_x, dt_y);
				const int physical_index = j * cuda::geometry.xsz + i;
				if (candidate < local_min || (candidate == local_min &&
					(local_index < 0 || physical_index < local_index)))
				{
					local_min = candidate;
					local_index = physical_index;
				}
			}
		}
	}

	shared_min[tid] = local_min;
	shared_index[tid] = local_index;
	__syncthreads();
	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if (tid < offset)
		{
			const NUMERIC_TYPE other = shared_min[tid + offset];
			const int other_index = shared_index[tid + offset];
			if (other < shared_min[tid] || (other == shared_min[tid] &&
				other_index >= 0 && (shared_index[tid] < 0 || other_index < shared_index[tid])))
			{
				shared_min[tid] = other;
				shared_index[tid] = other_index;
			}
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		const int out = blockIdx.y * gridDim.x + blockIdx.x;
		block_min[out] = shared_min[0];
		block_index[out] = shared_index[0];
	}
}

template<typename F>
__global__ void record_cfl_diagnostic
(
	const NUMERIC_TYPE* block_min,
	const int* block_index,
	int reduction_elements,
	F U,
	CflDiagnosticRecord* records,
	unsigned long long* record_count,
	int capacity
)
{
	__shared__ NUMERIC_TYPE shared_min[CUDA_BLOCK_SIZE];
	__shared__ int shared_index[CUDA_BLOCK_SIZE];
	const int tid = threadIdx.x;
	NUMERIC_TYPE local_min = cuda::solver_params.max_dt;
	int local_index = -1;
	for (int n = tid; n < reduction_elements; n += blockDim.x)
	{
		const NUMERIC_TYPE candidate = block_min[n];
		const int candidate_index = block_index[n];
		if (candidate < local_min || (candidate == local_min && candidate_index >= 0 &&
			(local_index < 0 || candidate_index < local_index)))
		{
			local_min = candidate;
			local_index = candidate_index;
		}
	}
	shared_min[tid] = local_min;
	shared_index[tid] = local_index;
	__syncthreads();
	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if (tid < offset)
		{
			const NUMERIC_TYPE other = shared_min[tid + offset];
			const int other_index = shared_index[tid + offset];
			if (other < shared_min[tid] || (other == shared_min[tid] && other_index >= 0 &&
				(shared_index[tid] < 0 || other_index < shared_index[tid])))
			{
				shared_min[tid] = other;
				shared_index[tid] = other_index;
			}
		}
		__syncthreads();
	}
	if (tid != 0) return;

	const unsigned long long slot = atomicAdd(record_count, 1ULL);
	if (slot >= static_cast<unsigned long long>(capacity)) return;
	CflDiagnosticRecord& rec = records[slot];
	rec.cell_index = shared_index[0];
	rec.limiting_axis = 0;
	rec.cfl_dt = shared_min[0];
	rec.H = rec.HU = rec.HV = C(0.0);
	rec.dt_x = rec.dt_y = cuda::solver_params.max_dt;
	rec.wave = C(0.0);
	for (int q = 0; q < 9; ++q)
	{
		rec.neighbour_H[q] = C(-1.0);
		rec.neighbour_HU[q] = C(0.0);
		rec.neighbour_HV[q] = C(0.0);
	}
	if (rec.cell_index < 0) return;

	const int i = rec.cell_index % cuda::geometry.xsz;
	const int j = rec.cell_index / cuda::geometry.xsz;
	const int k = (j + 1) * cuda::pitch + (i + 1);
	rec.H = U.H[k];
	rec.HU = U.HU[k];
	rec.HV = U.HV[k];
	if (rec.H > cuda::solver_params.DepthThresh)
	{
		const NUMERIC_TYPE inv_H = C(1.0) / rec.H;
		const NUMERIC_TYPE u = rec.HU * inv_H;
		const NUMERIC_TYPE v = rec.HV * inv_H;
		rec.wave = SQRT(cuda::physical_params.g * rec.H);
		rec.dt_x = cuda::solver_params.cfl * cuda::geometry.dx / (FABS(u) + rec.wave);
		rec.dt_y = cuda::solver_params.cfl * cuda::geometry.dy / (FABS(v) + rec.wave);
		rec.limiting_axis = rec.dt_x <= rec.dt_y ? 1 : 2;
	}
	int q = 0;
	for (int dj = -1; dj <= 1; ++dj)
	{
		for (int di = -1; di <= 1; ++di, ++q)
		{
			const int ni = i + di;
			const int nj = j + dj;
			if (ni < 0 || ni >= cuda::geometry.xsz || nj < 0 || nj >= cuda::geometry.ysz) continue;
			const int nk = (nj + 1) * cuda::pitch + (ni + 1);
			rec.neighbour_H[q] = U.H[nk];
			rec.neighbour_HU[q] = U.HU[nk];
			rec.neighbour_HV[q] = U.HV[nk];
		}
	}
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
reduction_elements(0),
diagnostic_index_field(nullptr),
diagnostic_records(nullptr),
diagnostic_count(nullptr),
diagnostic_capacity(0)
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
		if (diagnostic_records != nullptr)
		{
			lis::cuda::update_dt_block_min_diag<F><<<reduction_grid, CUDA_BLOCK_SIZE, 0, stream>>>(
					dt_field, diagnostic_index_field, U);
		}
		else
		{
			lis::cuda::update_dt_block_min<F><<<reduction_grid, CUDA_BLOCK_SIZE, 0, stream>>>(
					dt_field, U);
		}
		checkCudaErrors(cudaPeekAtLastError());
		checkCudaErrors(cub::DeviceReduce::Min(d_temp, bytes,
					dt_field, &dt, reduction_elements, stream));
		if (diagnostic_records != nullptr)
		{
			lis::cuda::record_cfl_diagnostic<F><<<1, CUDA_BLOCK_SIZE, 0, stream>>>(
					dt_field, diagnostic_index_field, reduction_elements, U,
					diagnostic_records, diagnostic_count, diagnostic_capacity);
			checkCudaErrors(cudaPeekAtLastError());
		}
	}
}

template<typename F>
void lis::cuda::DynamicTimestep<F>::enable_diagnostics(int capacity)
{
	if (!adaptive) return;
	if (capacity <= 0) capacity = 1;
	if (diagnostic_records != nullptr && diagnostic_capacity == capacity) return;
	disable_diagnostics();
	diagnostic_index_field = static_cast<int*>(malloc_device(reduction_elements * sizeof(int)));
	diagnostic_records = static_cast<CflDiagnosticRecord*>(malloc_device(
		static_cast<size_t>(capacity) * sizeof(CflDiagnosticRecord)));
	diagnostic_count = static_cast<unsigned long long*>(malloc_device(sizeof(unsigned long long)));
	diagnostic_capacity = capacity;
	checkCudaErrors(cudaMemset(diagnostic_count, 0, sizeof(unsigned long long)));
}

template<typename F>
void lis::cuda::DynamicTimestep<F>::disable_diagnostics()
{
	free_device(diagnostic_index_field);
	free_device(diagnostic_records);
	free_device(diagnostic_count);
	diagnostic_index_field = nullptr;
	diagnostic_records = nullptr;
	diagnostic_count = nullptr;
	diagnostic_capacity = 0;
}

template<typename F>
void lis::cuda::DynamicTimestep<F>::reset_diagnostics(cudaStream_t stream)
{
	if (diagnostic_count != nullptr)
		checkCudaErrors(cudaMemsetAsync(diagnostic_count, 0, sizeof(unsigned long long), stream));
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
		disable_diagnostics();
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