#include "ghostraster.h"
#include "cuda_activity.cuh"
#include "../swe/wet_dry.h"
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
__global__ void prepare_step_state
(
	F U,
	StepPreparation prep
)
{
	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	const int cells = cuda::geometry.xsz * cuda::geometry.ysz;
	if (k >= cells) return;
	if (prep.mask != nullptr && prep.mask[k] == 0) return;

	const int i = k % cuda::geometry.xsz;
	const int j = k / cuda::geometry.xsz;
	const int g = (j + 1) * cuda::pitch + (i + 1);

	if (prep.fixed_stage != nullptr && prep.dem != nullptr)
	{
		const NUMERIC_TYPE stage = prep.fixed_stage[k];
		if (isfinite(stage))
		{
			const NUMERIC_TYPE old_h = U.H[g];
			const NUMERIC_TYPE new_h = FMAX(C(0.0), stage - prep.dem[g]);
			U.H[g] = new_h;
			if (new_h == C(0.0))
			{
				U.HU[g] = C(0.0);
				U.HV[g] = C(0.0);
			}
			if (prep.boundary_volume != nullptr && prep.cell_area > C(0.0))
			{
				const NUMERIC_TYPE dv = (new_h - old_h) * prep.cell_area;
				if (dv != C(0.0))
					atomicAdd(prep.boundary_volume, static_cast<double>(dv));
			}
		}
	}

	if (U.H[g] < C(0.0))
	{
		if (prep.negative_volume != nullptr && prep.cell_area > C(0.0))
			atomicAdd(prep.negative_volume, -U.H[g] * prep.cell_area);
		U.H[g] = C(0.0);
		U.HU[g] = C(0.0);
		U.HV[g] = C(0.0);
	}
	else if (U.H[g] == C(0.0))
	{
		U.HU[g] = C(0.0);
		U.HV[g] = C(0.0);
	}

	if (prep.sample != nullptr && *prep.sample != 0)
	{
		const NUMERIC_TYPE h = U.H[g];
		if (prep.max_h != nullptr)
			prep.max_h[k] = FMAX(prep.max_h[k], h);
		if (prep.max_v != nullptr &&
			momentum_wet(h, U.HU[g], U.HV[g], cuda::solver_params.DepthThresh,
				cell_storage_depth(g)))
		{
			const NUMERIC_TYPE vx = U.HU[g] / h;
			const NUMERIC_TYPE vy = U.HV[g] / h;
			const NUMERIC_TYPE speed = SQRT(vx * vx + vy * vy);
			if (speed > prep.max_v[k])
			{
				prep.max_v[k] = speed;
				if (prep.peak_vx != nullptr) prep.peak_vx[k] = vx;
				if (prep.peak_vy != nullptr) prep.peak_vy[k] = vy;
			}
		}
	}
}

template<typename F>
__device__ NUMERIC_TYPE predicted_cfl_depth(F U,int g,int k,StepPreparation p)
{
    NUMERIC_TYPE h=U.H[g];
    if(!p.previous_cfl || p.growth_limit<=C(0.0)) return h;
    const NUMERIC_TYPE dtp=FMAX(C(0.0),*p.previous_cfl)*p.growth_limit;
    NUMERIC_TYPE rate=C(0.0);
    if(p.hydrology) rate+=p.hydrology[k];
    if(p.cell_source) rate+=p.cell_source[k];
    const int cls=p.mask ? p.mask[k] : 1;
    if(p.rainfall_rate>C(0.0) && cls!=2 &&
       (!p.dem || FABS(p.dem[g]-cuda::solver_params.nodata_elevation)>=C(1e-6)))
        rate+=p.rainfall_rate;
    if(p.point_head && p.point_next) {
        const NUMERIC_TYPE area=p.cell_area>C(0.0) ? p.cell_area :
            cuda::geometry.dx*cuda::geometry.dy;
        for(int n=p.point_head[k];n>=0;n=p.point_next[n]) {
            const ESourceType type=cuda::boundaries.PS_type[n];
            if(type==QFIX4 || type==QVAR5)
                rate+=cuda::boundaries.PS_value[n]*cuda::geometry.dx/area;
        }
    }
    return FMAX(C(0.0),h+dtp*rate);
}

template<typename F>
__global__ void update_dt_block_min
(
	NUMERIC_TYPE* block_min,
	F U,
    StepPreparation prep
)
{
	__shared__ NUMERIC_TYPE shared_min[CUDA_BLOCK_SIZE];
	const int tid = threadIdx.x;
	const int i = blockIdx.x * blockDim.x + tid;
	NUMERIC_TYPE local_min = cuda::solver_params.max_dt;
    if(prep.previous_cfl && prep.growth_limit>C(0.0))
        local_min=FMIN(local_min,*prep.previous_cfl*prep.growth_limit);

	if (i < cuda::geometry.xsz)
	{
		for (int j = blockIdx.y; j < cuda::geometry.ysz; j += gridDim.y)
		{
			const int k = (j + 1) * cuda::pitch + (i + 1);
			const NUMERIC_TYPE H = predicted_cfl_depth(U,k,j*cuda::geometry.xsz+i,prep);
			if (momentum_wet(H, U.HU[k], U.HV[k], cuda::solver_params.DepthThresh, cell_storage_depth(k)))
			{
				const NUMERIC_TYPE inv_H = C(1.0) / H;
				const NUMERIC_TYPE u = U.HU[k] * inv_H;
				const NUMERIC_TYPE v = U.HV[k] * inv_H;
				const NUMERIC_TYPE wave = SQRT(cuda::physical_params.g * H);
				const NUMERIC_TYPE rate_x = (FABS(u) + wave) / cuda::geometry.dx;
				const NUMERIC_TYPE rate_y = (FABS(v) + wave) / cuda::geometry.dy;
				local_min = FMIN(local_min, cuda::solver_params.cfl / (rate_x + rate_y));
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
	F U,
    StepPreparation prep
)
{
	__shared__ NUMERIC_TYPE shared_min[CUDA_BLOCK_SIZE];
	__shared__ int shared_index[CUDA_BLOCK_SIZE];
	const int tid = threadIdx.x;
	const int i = blockIdx.x * blockDim.x + tid;
	NUMERIC_TYPE local_min = cuda::solver_params.max_dt;
    if(prep.previous_cfl && prep.growth_limit>C(0.0))
        local_min=FMIN(local_min,*prep.previous_cfl*prep.growth_limit);
	int local_index = -1;

	if (i < cuda::geometry.xsz)
	{
		for (int j = blockIdx.y; j < cuda::geometry.ysz; j += gridDim.y)
		{
			const int k = (j + 1) * cuda::pitch + (i + 1);
			const NUMERIC_TYPE H = predicted_cfl_depth(U,k,j*cuda::geometry.xsz+i,prep);
			if (momentum_wet(H, U.HU[k], U.HV[k], cuda::solver_params.DepthThresh, cell_storage_depth(k)))
			{
				const NUMERIC_TYPE inv_H = C(1.0) / H;
				const NUMERIC_TYPE u = U.HU[k] * inv_H;
				const NUMERIC_TYPE v = U.HV[k] * inv_H;
				const NUMERIC_TYPE wave = SQRT(cuda::physical_params.g * H);
				const NUMERIC_TYPE rate_x = (FABS(u) + wave) / cuda::geometry.dx;
				const NUMERIC_TYPE rate_y = (FABS(v) + wave) / cuda::geometry.dy;
				const NUMERIC_TYPE candidate = cuda::solver_params.cfl / (rate_x + rate_y);
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
	int capacity,
    StepPreparation prep
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
	rec.H = predicted_cfl_depth(U,k,rec.cell_index,prep);
	rec.HU = U.HU[k];
	rec.HV = U.HV[k];
	if (momentum_wet(rec.H, rec.HU, rec.HV, cuda::solver_params.DepthThresh, cell_storage_depth(k)))
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
void lis::cuda::DynamicTimestep<F>::update_dt_async
(
	cudaStream_t stream,
	const StepPreparation& preparation
)
{
	auto& U = solver.d_U();
	const int cells = xsz * ysz;
	const int blocks = std::max(1, (cells + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
	lis::cuda::prepare_step_state<F><<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(
		U, preparation);
	checkCudaErrors(cudaPeekAtLastError());
	reduce_dt_async(stream, preparation);
}

template<typename F>
void lis::cuda::DynamicTimestep<F>::update_dt_async(cudaStream_t stream)
{
    reduce_dt_async(stream, StepPreparation{});
}

template<typename F>
void lis::cuda::DynamicTimestep<F>::reduce_dt_async(cudaStream_t stream,const StepPreparation& preparation)
{
	if (adaptive)
	{
		auto& U = solver.d_U();
		if (diagnostic_records != nullptr)
		{
			lis::cuda::update_dt_block_min_diag<F><<<reduction_grid, CUDA_BLOCK_SIZE, 0, stream>>>(
					dt_field, diagnostic_index_field, U, preparation);
		}
		else
		{
			lis::cuda::update_dt_block_min<F><<<reduction_grid, CUDA_BLOCK_SIZE, 0, stream>>>(
					dt_field, U, preparation);
		}
		checkCudaErrors(cudaPeekAtLastError());
		checkCudaErrors(cub::DeviceReduce::Min(d_temp, bytes,
					dt_field, &dt, reduction_elements, stream));
		if (diagnostic_records != nullptr)
		{
			lis::cuda::record_cfl_diagnostic<F><<<1, CUDA_BLOCK_SIZE, 0, stream>>>(
					dt_field, diagnostic_index_field, reduction_elements, U,
					diagnostic_records, diagnostic_count, diagnostic_capacity, preparation);
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