#pragma once
#include "../lisflood.h"
#include "params.h"
#include "stats.h"
#include "fv1/cuda_fv1_flow.cuh"
#include "cuda_sample.cuh"
#include "cuda_solver.cuh"

namespace lis
{
namespace cuda
{
namespace fv1
{

enum SparseFaceType
{
	SPARSE_FACE_TRANSMISSIVE_OUTFLOW = 1,
	SPARSE_FACE_FIXED_FLUX = 2,
	SPARSE_FACE_WEIR = 3
};

struct SparseFace
{
	int neg_g;
	int pos_g;
	int face_g;
	int neg_cell;
	int pos_cell;
	int axis; /* 0=x: west->east, 1=y: south->north */
	int type;
	NUMERIC_TYPE p0;
	NUMERIC_TYPE p1;
	NUMERIC_TYPE p2;
	NUMERIC_TYPE p3;
};

struct SparseFaceFlux
{
	FlowVector base;
	FlowVector desired;
};

class Solver : public cuda::Solver<Flow>
{
public:
	Solver
	(
		Flow& U,
		NUMERIC_TYPE* DEM,
		NUMERIC_TYPE* Zstar_x,
		NUMERIC_TYPE* Zstar_y,
		NUMERIC_TYPE* manning,
		Geometry& geometry,
		PhysicalParams& physical_params,
		dim3 grid_size
	);

	void FloodplainQ(); 
	void zero_thin_depth_slopes();

	void zero_ghost_cells();

	void update_ghost_cells() { update_ghost_cells(0); }
	void update_ghost_cells(cudaStream_t stream);

	void clamp_negative_depths() { clamp_negative_depths(0); }
	void clamp_negative_depths(cudaStream_t stream);

	void set_sparse_faces(const SparseFace* faces, int count);
	
	void update_uniform_rain(NUMERIC_TYPE rain_rate) { update_uniform_rain(rain_rate, 0); }
	void update_uniform_rain(NUMERIC_TYPE rain_rate, cudaStream_t stream);

	void updateMaxFieldACC(NUMERIC_TYPE t);

	Flow& update_flow_variables
	(
		MassStats* mass_stats
	) { return update_flow_variables(mass_stats, 0); }
	Flow& update_flow_variables
	(
		MassStats* mass_stats,
		cudaStream_t stream
	);

	Flow& d_U();

	void update_dt_per_element
	(
		NUMERIC_TYPE* dt_field
	) const;

	void swap_state();

	void set_negative_depth_counter(NUMERIC_TYPE* counter)
	{
		negative_depth_volume = counter;
	}

	~Solver();

private:
	Flow U1;
	Flow U2;
	Flow Ux;
	Flow& Uold;
	Flow& U;
	NUMERIC_TYPE* DEM;
	NUMERIC_TYPE* Zstar_x;
	NUMERIC_TYPE* Zstar_y;
	NUMERIC_TYPE* manning;
	NUMERIC_TYPE* negative_depth_volume;
	const SparseFace* sparse_faces;
	SparseFaceFlux* sparse_fluxes;
	int sparse_face_count;
	int sparse_flux_capacity;
	bool friction;
	const dim3 grid_size;
};

}
}
}
