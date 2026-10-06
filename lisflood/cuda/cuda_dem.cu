#include "cuda_dem.cuh"
#include "cuda_solver.cuh"
#include "io.h"

namespace lis
{
namespace cuda
{
namespace fv1
{

__global__ void initialise_Zstar_x
(
	NUMERIC_TYPE* __restrict__ Zstar_x,
	const NUMERIC_TYPE* __restrict__ DEM
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;

	for (int j=global_j+1; j<cuda::geometry.ysz+1; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i; i<cuda::geometry.xsz+1; i+=blockDim.x*gridDim.x)
		{
			NUMERIC_TYPE Z_neg = DEM[j*cuda::pitch + i];
			NUMERIC_TYPE Z_pos = DEM[j*cuda::pitch + i+1];

			Zstar_x[j*cuda::pitch + i] = FMAX(Z_neg, Z_pos);
		}
	}
}

__global__ void initialise_Zstar_y
(
	NUMERIC_TYPE* __restrict__ Zstar_y,
	const NUMERIC_TYPE* __restrict__ DEM
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;

	for (int j=global_j; j<cuda::geometry.ysz+1; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i+1; i<cuda::geometry.xsz+1; i+=blockDim.x*gridDim.x)
		{
			NUMERIC_TYPE Z_neg = DEM[(j+1)*cuda::pitch + i];
			NUMERIC_TYPE Z_pos = DEM[j*cuda::pitch + i];

			Zstar_y[j*cuda::pitch + i] = FMAX(Z_neg, Z_pos);
		}
	}
}

__global__ void apply_mask_to_Zstar_x
(
	NUMERIC_TYPE* __restrict__ Zstar_x,
	const NUMERIC_TYPE* __restrict__ DEM,
	const int* __restrict__ cell_mask
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;

	for (int j=global_j+1; j<cuda::geometry.ysz+1; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i+1; i<cuda::geometry.xsz; i+=blockDim.x*gridDim.x)
		{
			const int neg_k = (j-1)*cuda::geometry.xsz + (i-1);
			const int pos_k = neg_k + 1;
			const bool neg_active = cell_mask[neg_k] != 0;
			const bool pos_active = cell_mask[pos_k] != 0;
			if (neg_active == pos_active) continue;

			const int g = j*cuda::pitch + i;
			Zstar_x[g] = neg_active ? DEM[g] : DEM[g+1];
		}
	}
}

__global__ void apply_mask_to_Zstar_y
(
	NUMERIC_TYPE* __restrict__ Zstar_y,
	const NUMERIC_TYPE* __restrict__ DEM,
	const int* __restrict__ cell_mask
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;

	for (int j=global_j+1; j<cuda::geometry.ysz; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i+1; i<cuda::geometry.xsz+1; i+=blockDim.x*gridDim.x)
		{
			const int pos_k = (j-1)*cuda::geometry.xsz + (i-1);
			const int neg_k = pos_k + cuda::geometry.xsz;
			const bool pos_active = cell_mask[pos_k] != 0;
			const bool neg_active = cell_mask[neg_k] != 0;
			if (neg_active == pos_active) continue;

			const int g = j*cuda::pitch + i;
			Zstar_y[g] = neg_active ? DEM[g+cuda::pitch] : DEM[g];
		}
	}
}


}
}
}

NUMERIC_TYPE* lis::cuda::Topography::load
(
	const char* filename,
	Geometry& geometry,
	int& pitch,
	int& offset,
	NUMERIC_TYPE nodata_elevation,
	int acceleration,
	int verbose
)
{
	FILE* dem_file = fopen_or_die(filename, "rb", "Loading DEM\n", verbose);
	NUMERIC_TYPE original_no_data_value = C(0.0);
	AsciiRaster::read_header(dem_file, geometry, original_no_data_value);
	NUMERIC_TYPE* DEM; 

    if (acceleration == 0){ 
		DEM = lis::GhostRaster::allocate(geometry); 
	    pitch = lis::GhostRaster::pitch(geometry);
	    offset = lis::GhostRaster::offset(geometry);
	} else {                                            
		DEM = lis::GhostRaster::allocate_H(geometry); 
	    pitch = lis::GhostRaster::pitch_ACC(geometry);   
	    offset = lis::GhostRaster::offset_ACC(geometry);	
	}
	AsciiRaster::read(dem_file, DEM, geometry, pitch, offset); 
	AsciiRaster::replace_no_data(DEM, geometry, pitch, offset,
			original_no_data_value, nodata_elevation);

	fclose(dem_file);


	return DEM;
}

void lis::cuda::Topography::initialise_Zstar_x
(
	NUMERIC_TYPE* __restrict__ Zstar_x,
	const NUMERIC_TYPE* __restrict__ DEM
)
{
	lis::cuda::fv1::initialise_Zstar_x<<<1, cuda::block_size>>>(Zstar_x, DEM);
}

void lis::cuda::Topography::initialise_Zstar_y
(
	NUMERIC_TYPE* __restrict__ Zstar_y,
	const NUMERIC_TYPE* __restrict__ DEM
)
{
	lis::cuda::fv1::initialise_Zstar_y<<<1, cuda::block_size>>>(Zstar_y, DEM);
}

void lis::cuda::Topography::initialise_Zstar_x
(
	NUMERIC_TYPE* __restrict__ Zstar_x,
	const NUMERIC_TYPE* __restrict__ DEM,
	const int* cell_mask
)
{
	initialise_Zstar_x(Zstar_x, DEM);
	if (cell_mask != nullptr)
		lis::cuda::fv1::apply_mask_to_Zstar_x<<<1, cuda::block_size>>>(
			Zstar_x, DEM, cell_mask);
}

void lis::cuda::Topography::initialise_Zstar_y
(
	NUMERIC_TYPE* __restrict__ Zstar_y,
	const NUMERIC_TYPE* __restrict__ DEM,
	const int* cell_mask
)
{
	initialise_Zstar_y(Zstar_y, DEM);
	if (cell_mask != nullptr)
		lis::cuda::fv1::apply_mask_to_Zstar_y<<<1, cuda::block_size>>>(
			Zstar_y, DEM, cell_mask);
}

void lis::cuda::Topography::clamp_boundary_values
(
	NUMERIC_TYPE* DEM,
	Geometry& geometry,
	int pitch
)
{
	for (int j=1; j<geometry.ysz+1; j++)
	{
		// west
		{
			const int i=0;
			DEM[j*pitch + i] = DEM[j*pitch + i+1];
		}

		// east
		{
			const int i=geometry.xsz+1;
			//if (j == 161 || j == 162 || j == 163 || j == 164) {
			//	DEM[j * pitch + i] = -C(20.5);
			//}
			//else
			//{
				DEM[j * pitch + i] = DEM[j * pitch + i - 1];
//			}
		}
	}

	for (int i=1; i<geometry.xsz+1; i++)
	{
		// north
		{
			const int j=0;
			DEM[j*pitch + i] = DEM[(j+1)*pitch + i];
		}

		// south
		{
			const int j=geometry.ysz+1;
			DEM[j*pitch + i] = DEM[(j-1)*pitch + i];
		}
	}
}
