#pragma once
#define CUDA_BLOCK_SIZE_X 16
#define CUDA_BLOCK_SIZE_Y 16
#define CUDA_BLOCK_SIZE 256

#include "../geometry.h"
#include "cuda_boundary.cuh"
#include "cuda_stats.cuh"
#include "params.h"
#include "sample.h"

namespace lis
{
namespace cuda
{

extern __constant__ Geometry geometry;
extern __constant__ Geometry rain_geometry;
extern __constant__ int pitch;
extern __constant__ PhysicalParams physical_params;
extern __constant__ SolverParams solver_params;
extern __constant__ BoundaryConditions boundaries;
extern __constant__ SamplePoints sample_points;
extern __managed__ NUMERIC_TYPE dt;
extern __managed__ int sample_buf_idx;

extern const dim3 block_size;

struct CflDiagnosticRecord
{
	int cell_index;
	int limiting_axis; /* 0=none/dry, 1=x, 2=y */
	NUMERIC_TYPE cfl_dt;
	NUMERIC_TYPE H;
	NUMERIC_TYPE HU;
	NUMERIC_TYPE HV;
	NUMERIC_TYPE dt_x;
	NUMERIC_TYPE dt_y;
	NUMERIC_TYPE wave;
	NUMERIC_TYPE neighbour_H[9];
	NUMERIC_TYPE neighbour_HU[9];
	NUMERIC_TYPE neighbour_HV[9];
};

__device__
void update_mass_stats_x
(
	MassStats* stats,
	NUMERIC_TYPE FH,
	int i,
	int j,
	NUMERIC_TYPE time_stage_fraction = C(1.0)
);

__device__
void update_mass_stats_y
(
	MassStats* stats,
	NUMERIC_TYPE FH,
	int i,
	int j,
	NUMERIC_TYPE time_stage_fraction = C(1.0)
);

template<typename F>
class Solver
{
	public:
	virtual void zero_ghost_cells() = 0;

	virtual void update_ghost_cells() = 0;

	virtual void update_uniform_rain(NUMERIC_TYPE rain_rate) = 0;

	virtual void updateMaxFieldACC(NUMERIC_TYPE t) = 0;
	
	virtual void zero_thin_depth_slopes() = 0;

	virtual void FloodplainQ() = 0; 

	virtual F& update_flow_variables
	(
		MassStats* mass_stats
	) = 0;

	virtual F& d_U() = 0;

	virtual void update_dt_per_element
	(
		NUMERIC_TYPE* dt_field
	) const = 0;

	virtual ~Solver() {};
};

template<typename F>
class DynamicTimestep
{
public:
	DynamicTimestep
	(
	 	NUMERIC_TYPE& dt,
		Geometry& geometry,
		NUMERIC_TYPE max_dt,
		int adaptive_ts,
		Solver<F>& solver
	);

	void update_dt_async(cudaStream_t stream = 0);
	NUMERIC_TYPE update_dt();

	void enable_diagnostics(int capacity);
	void disable_diagnostics();
	void reset_diagnostics(cudaStream_t stream = 0);
	CflDiagnosticRecord* diagnostics_records_device() const { return diagnostic_records; }
	unsigned long long* diagnostics_count_device() const { return diagnostic_count; }
	int diagnostics_capacity() const { return diagnostic_capacity; }
	bool diagnostics_enabled() const { return diagnostic_records != nullptr; }

	~DynamicTimestep();

private:
	bool adaptive;
	NUMERIC_TYPE& dt;
	NUMERIC_TYPE max_dt;
	Solver<F>& solver;
	NUMERIC_TYPE* dt_field;
	const int xsz;
	const int ysz;
	dim3 reduction_grid;
	int reduction_elements;
	void* d_temp;
	size_t bytes;
	int* diagnostic_index_field;
	CflDiagnosticRecord* diagnostic_records;
	unsigned long long* diagnostic_count;
	int diagnostic_capacity;
};


template<typename F>
class DynamicTimestepACC
{
public:
	DynamicTimestepACC
	(
		NUMERIC_TYPE& dt,
		Geometry& geometry,
		NUMERIC_TYPE max_dt,
		int adaptive_ts,
		Solver<F>& solver
	);

	NUMERIC_TYPE update_dt_ACC();

	~DynamicTimestepACC();

private:
	bool adaptive;
	NUMERIC_TYPE& dt;
	NUMERIC_TYPE max_dt;
	const Solver<F>& solver;
	NUMERIC_TYPE* dt_field;
	const int elements;
	void* d_temp;
	size_t bytes;
};


}
}

#include "cuda_solver.templates.cu"