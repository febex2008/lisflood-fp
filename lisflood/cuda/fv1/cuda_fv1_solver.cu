#include "cuda_fv1_solver.cuh"
#include "cuda_boundary.cuh"
#include "cuda_hll.cuh"
#include "cuda_solver.cuh"
#include <algorithm>

namespace lis
{
namespace cuda
{
namespace fv1
{

__global__ void zero_ghost_cells_north_south
(
	Flow U
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;

	for (int i=global_i; i<cuda::pitch; i+=blockDim.x*gridDim.x)
	{
		{
			int j=0;
			U.H[j*pitch + i] = C(0.0);
			U.HU[j*pitch + i] = C(0.0);
			U.HV[j*pitch + i] = C(0.0);
		}

		{
			int j=cuda::geometry.ysz+1;
			U.H[j*pitch + i] = C(0.0);
			U.HU[j*pitch + i] = C(0.0);
			U.HV[j*pitch + i] = C(0.0);
		}
	}
}

__global__ void zero_ghost_cells_east_west
(
	Flow U
)
{
	int global_j = blockIdx.x*blockDim.x + threadIdx.x;

	for (int j=global_j; j<cuda::geometry.ysz+2; j+=blockDim.x*gridDim.x)
	{
		{
			int i=0;
			U.H[j*pitch + i] = C(0.0);
			U.HU[j*pitch + i] = C(0.0);
			U.HV[j*pitch + i] = C(0.0);
		}

		{
			int i=cuda::geometry.xsz+1;
			U.H[j*pitch + i] = C(0.0);
			U.HU[j*pitch + i] = C(0.0);
			U.HV[j*pitch + i] = C(0.0);
		}
	}
}

__device__ inline bool active_cell(const int* cell_mask, int i, int j)
{
	if (i < 1 || i > cuda::geometry.xsz || j < 1 || j > cuda::geometry.ysz) return false;
	if (cell_mask == nullptr) return true;
	const int k = (j-1)*cuda::geometry.xsz + (i-1);
	return cell_mask[k] != 0;
}

__global__ void apply_mask_to_zstar_x
(
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE* Zstar_x,
	const int* cell_mask
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;
	for (int j=global_j+1; j<=cuda::geometry.ysz; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i+1; i<cuda::geometry.xsz; i+=blockDim.x*gridDim.x)
		{
			const bool neg_active = active_cell(cell_mask, i, j);
			const bool pos_active = active_cell(cell_mask, i+1, j);
			if (neg_active == pos_active) continue;
			const int g = j*cuda::pitch + i;
			Zstar_x[g] = neg_active ? DEM[g] : DEM[g+1];
		}
	}
}

__global__ void apply_mask_to_zstar_y
(
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE* Zstar_y,
	const int* cell_mask
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;
	for (int j=global_j+1; j<cuda::geometry.ysz; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i+1; i<=cuda::geometry.xsz; i+=blockDim.x*gridDim.x)
		{
			const bool pos_active = active_cell(cell_mask, i, j);
			const bool neg_active = active_cell(cell_mask, i, j+1);
			if (neg_active == pos_active) continue;
			const int g = j*cuda::pitch + i;
			Zstar_y[g] = neg_active ? DEM[g+cuda::pitch] : DEM[g];
		}
	}
}

__global__ void clamp_negative_depths_kernel
(
	Flow U,
	NUMERIC_TYPE* negative_depth_volume,
	const int* cell_mask
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;
	for (int j=global_j+1; j<cuda::geometry.ysz+1; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i+1; i<cuda::geometry.xsz+1; i+=blockDim.x*gridDim.x)
		{
			const int k = j*cuda::pitch + i;
			if (!active_cell(cell_mask, i, j)) continue;
			NUMERIC_TYPE& H = U.H[k];
			NUMERIC_TYPE& HU = U.HU[k];
			NUMERIC_TYPE& HV = U.HV[k];
			if (H < C(0.0))
			{
				const NUMERIC_TYPE correction = -H * cuda::geometry.dx * cuda::geometry.dy;
				H = C(0.0);
				HU = C(0.0);
				HV = C(0.0);
				if (negative_depth_volume != nullptr) atomicAdd(negative_depth_volume, correction);
			}
			else if (H == C(0.0))
			{
				HU = C(0.0);
				HV = C(0.0);
			}
		}
	}
}

__device__ inline FlowVector sparse_hll
(
	int axis,
	const FlowVector& U_neg,
	const FlowVector& U_pos
)
{
	return axis == 0 ? HLL::x(U_neg, U_pos) : HLL::y(U_neg, U_pos);
}

__device__ inline NUMERIC_TYPE weir_blocked_pressure
(
	NUMERIC_TYPE H,
	NUMERIC_TYPE Z,
	NUMERIC_TYPE crest
)
{
	if (H <= C(0.0)) return C(0.0);
	const NUMERIC_TYPE blocked = FMIN(H, FMAX(C(0.0), crest - Z));
	return cuda::physical_params.g * (H * blocked - C(0.5) * blocked * blocked);
}

__global__ void prepare_sparse_face_fluxes
(
	Flow Uold,
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE* Zstar_x,
	NUMERIC_TYPE* Zstar_y,
	const SparseFace* faces,
	SparseFaceFlux* fluxes,
	int count
)
{
	for (int n=blockIdx.x*blockDim.x+threadIdx.x; n<count; n+=blockDim.x*gridDim.x)
	{
		const SparseFace face = faces[n];
		const bool boundary_face =
			face.type == SPARSE_FACE_TRANSMISSIVE_OUTFLOW ||
			face.type == SPARSE_FACE_FREE || face.type == SPARSE_FACE_CLOSED;
		const bool neg_inside = face.neg_cell >= 0;
		const bool pos_inside = face.pos_cell >= 0;
		const bool masked_boundary = boundary_face && (neg_inside != pos_inside) && face.p3 > C(0.5);

		if (masked_boundary)
		{
			SparseFaceFlux result;
			result.base = { C(0.0), C(0.0), C(0.0) };
			FlowVector U_inside = neg_inside ? Uold[face.neg_g] : Uold[face.pos_g];
			FlowVector U_outside = U_inside;
			if (face.type == SPARSE_FACE_CLOSED)
			{
				if (face.axis == 0) U_outside.HU = -U_outside.HU;
				else U_outside.HV = -U_outside.HV;
			}
			else if (face.type == SPARSE_FACE_TRANSMISSIVE_OUTFLOW)
			{
				const NUMERIC_TYPE normal_momentum = face.axis == 0 ? U_inside.HU : U_inside.HV;
				const NUMERIC_TYPE outward_momentum = (neg_inside ? C(1.0) : C(-1.0)) * normal_momentum;
				if (outward_momentum < C(0.0))
				{
					if (face.axis == 0) U_outside.HU = -U_outside.HU;
					else U_outside.HV = -U_outside.HV;
				}
			}
			const FlowVector desired = neg_inside
				? sparse_hll(face.axis, U_inside, U_outside)
				: sparse_hll(face.axis, U_outside, U_inside);
			result.desired_neg = desired;
			result.desired_pos = desired;
			fluxes[n] = result;
			continue;
		}

		const NUMERIC_TYPE Zstar = face.axis == 0 ? Zstar_x[face.face_g] : Zstar_y[face.face_g];
		const FlowVector U_neg = Uold[face.neg_g];
		const FlowVector U_pos = Uold[face.pos_g];
		const FlowVector Ustar_neg = U_neg.star(DEM[face.neg_g], Zstar);
		const FlowVector Ustar_pos = U_pos.star(DEM[face.pos_g], Zstar);
		SparseFaceFlux result;
		result.base = sparse_hll(face.axis, Ustar_neg, Ustar_pos);
		result.desired_neg = result.base;
		result.desired_pos = result.base;

		if (boundary_face)
		{
			if (neg_inside != pos_inside)
			{
				FlowVector U_inside = neg_inside ? Ustar_neg : Ustar_pos;
				FlowVector U_outside = U_inside;
				if (face.type == SPARSE_FACE_CLOSED)
				{
					if (face.axis == 0) U_outside.HU = -U_outside.HU;
					else U_outside.HV = -U_outside.HV;
				}
				else if (face.type == SPARSE_FACE_TRANSMISSIVE_OUTFLOW)
				{
					const NUMERIC_TYPE normal_momentum = face.axis == 0 ? U_inside.HU : U_inside.HV;
					const NUMERIC_TYPE outward_momentum = (neg_inside ? C(1.0) : C(-1.0)) * normal_momentum;
					if (outward_momentum < C(0.0))
					{
						if (face.axis == 0) U_outside.HU = -U_outside.HU;
						else U_outside.HV = -U_outside.HV;
					}
				}
				const FlowVector desired = neg_inside
					? sparse_hll(face.axis, U_inside, U_outside)
					: sparse_hll(face.axis, U_outside, U_inside);
				result.desired_neg = desired;
				result.desired_pos = desired;
			}
		}
		else if (face.type == SPARSE_FACE_FIXED_FLUX)
		{
			const FlowVector desired = { face.p0, face.p1, face.p2 };
			result.desired_neg = desired;
			result.desired_pos = desired;
		}
		else if (face.type == SPARSE_FACE_WEIR && face.neg_cell >= 0 && face.pos_cell >= 0)
		{
			const NUMERIC_TYPE Z_neg = DEM[face.neg_g];
			const NUMERIC_TYPE Z_pos = DEM[face.pos_g];
			const NUMERIC_TYPE eta_neg = Z_neg + U_neg.H;
			const NUMERIC_TYPE eta_pos = Z_pos + U_pos.H;
			const bool neg_upstream = eta_neg >= eta_pos;
			const NUMERIC_TYPE eta_up = neg_upstream ? eta_neg : eta_pos;
			const NUMERIC_TYPE eta_down = neg_upstream ? eta_pos : eta_neg;
			const FlowVector U_up = neg_upstream ? U_neg : U_pos;
			const NUMERIC_TYPE crest = face.p0;
			const NUMERIC_TYPE cw = FMAX(C(0.0), face.p1);
			const NUMERIC_TYPE exponent = face.p2 > C(0.0) ? face.p2 : C(0.385);
			const NUMERIC_TYPE width_fraction = FMIN(C(1.0), FMAX(C(0.0), face.p3));
			const NUMERIC_TYPE open_fraction = C(1.0) - width_fraction;
			const NUMERIC_TYPE h_up = FMAX(C(0.0), eta_up - crest);
			const NUMERIC_TYPE h_down = FMAX(C(0.0), eta_down - crest);
			NUMERIC_TYPE q_weir = C(0.0);
			if (width_fraction > C(0.0) && h_up > cuda::solver_params.DepthThresh && cw > C(0.0))
			{
				NUMERIC_TYPE qmag = width_fraction * cw * POW(h_up, C(1.5));
				if (h_down > C(0.0))
				{
					const NUMERIC_TYPE ratio = FMIN(C(1.0), h_down / h_up);
					const NUMERIC_TYPE sub = FMAX(C(0.0), C(1.0) - POW(ratio, C(1.5)));
					qmag *= POW(sub, exponent);
				}
				q_weir = neg_upstream ? qmag : -qmag;
			}

			NUMERIC_TYPE tangential_velocity = C(0.0);
			if (U_up.H > cuda::solver_params.DepthThresh)
				tangential_velocity = face.axis == 0 ? U_up.HV / U_up.H : U_up.HU / U_up.H;
			const NUMERIC_TYPE active_h = FMAX(h_up, cuda::solver_params.DepthThresh);
			const NUMERIC_TYPE normal_advective = width_fraction > C(0.0)
				? q_weir * q_weir / (width_fraction * active_h) : C(0.0);
			const NUMERIC_TYPE wall_neg = width_fraction * weir_blocked_pressure(U_neg.H, Z_neg, crest);
			const NUMERIC_TYPE wall_pos = width_fraction * weir_blocked_pressure(U_pos.H, Z_pos, crest);
			const NUMERIC_TYPE mass_flux = open_fraction * result.base.H + q_weir;
			const NUMERIC_TYPE tangential_flux = open_fraction *
				(face.axis == 0 ? result.base.HV : result.base.HU) + q_weir * tangential_velocity;
			const NUMERIC_TYPE normal_neg = open_fraction *
				(face.axis == 0 ? result.base.HU : result.base.HV) + wall_neg + normal_advective;
			const NUMERIC_TYPE normal_pos = open_fraction *
				(face.axis == 0 ? result.base.HU : result.base.HV) + wall_pos + normal_advective;

			result.desired_neg.H = mass_flux;
			result.desired_pos.H = mass_flux;
			if (face.axis == 0)
			{
				result.desired_neg.HU = normal_neg;
				result.desired_pos.HU = normal_pos;
				result.desired_neg.HV = tangential_flux;
				result.desired_pos.HV = tangential_flux;
			}
			else
			{
				result.desired_neg.HU = tangential_flux;
				result.desired_pos.HU = tangential_flux;
				result.desired_neg.HV = normal_neg;
				result.desired_pos.HV = normal_pos;
			}
		}
		fluxes[n] = result;
	}
}

__global__ void apply_sparse_face_fluxes
(
	Flow U,
	const SparseFace* faces,
	const SparseFaceFlux* fluxes,
	int count,
	MassStats* mass_stats
)
{
	for (int n=blockIdx.x*blockDim.x+threadIdx.x; n<count; n+=blockDim.x*gridDim.x)
	{
		const SparseFace face = faces[n];
		const SparseFaceFlux pair = fluxes[n];
		const NUMERIC_TYPE scale = cuda::dt / (face.axis == 0 ? cuda::geometry.dx : cuda::geometry.dy);
		if (face.neg_cell >= 0)
		{
			const FlowVector dF = pair.desired_neg - pair.base;
			atomicAdd(&U.H[face.neg_g], -scale * dF.H);
			atomicAdd(&U.HU[face.neg_g], -scale * dF.HU);
			atomicAdd(&U.HV[face.neg_g], -scale * dF.HV);
		}
		if (face.pos_cell >= 0)
		{
			const FlowVector dF = pair.desired_pos - pair.base;
			atomicAdd(&U.H[face.pos_g], scale * dF.H);
			atomicAdd(&U.HU[face.pos_g], scale * dF.HU);
			atomicAdd(&U.HV[face.pos_g], scale * dF.HV);
		}

		if (mass_stats != nullptr && ((face.neg_cell >= 0) != (face.pos_cell >= 0)))
		{
			const bool neg_inside = face.neg_cell >= 0;
			const NUMERIC_TYPE outward_sign = neg_inside ? C(1.0) : C(-1.0);
			const NUMERIC_TYPE edge_length = face.axis == 0 ? cuda::geometry.dy : cuda::geometry.dx;
			const NUMERIC_TYPE base_outward = outward_sign * pair.base.H * edge_length;
			const NUMERIC_TYPE desired_mass = neg_inside ? pair.desired_neg.H : pair.desired_pos.H;
			const NUMERIC_TYPE desired_outward = outward_sign * desired_mass * edge_length;
			const NUMERIC_TYPE base_out = FMAX(C(0.0), base_outward);
			const NUMERIC_TYPE base_in = FMAX(C(0.0), -base_outward);
			const NUMERIC_TYPE desired_out = FMAX(C(0.0), desired_outward);
			const NUMERIC_TYPE desired_in = FMAX(C(0.0), -desired_outward);
			const bool internal_domain =
				(face.type == SPARSE_FACE_TRANSMISSIVE_OUTFLOW ||
				 face.type == SPARSE_FACE_FREE || face.type == SPARSE_FACE_CLOSED) &&
				face.p3 > C(0.5);
			if (internal_domain)
			{
				atomicAdd(&(mass_stats->out), desired_out);
				atomicAdd(&(mass_stats->in), desired_in);
			}
			else
			{
				atomicAdd(&(mass_stats->out), desired_out - base_out);
				atomicAdd(&(mass_stats->in), desired_in - base_in);
			}
		}
	}
}

__global__ void update_dt_per_element
(
	NUMERIC_TYPE* dt,
	Flow U,
	const int* cell_mask
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;

	for (int j=global_j; j<cuda::geometry.ysz+2; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i; i<cuda::pitch; i+=blockDim.x*gridDim.x)
		{
			if (cell_mask != nullptr && !active_cell(cell_mask, i, j))
			{
				dt[j*cuda::pitch + i] = cuda::solver_params.max_dt;
				continue;
			}
			NUMERIC_TYPE H = U.H[j*cuda::pitch + i];

			if (H > cuda::solver_params.DepthThresh)
			{
				NUMERIC_TYPE HU = U.HU[j*cuda::pitch + i];
				NUMERIC_TYPE HV = U.HV[j*cuda::pitch + i];
				NUMERIC_TYPE U = HU/H;
				NUMERIC_TYPE V = HV/H;
				
				NUMERIC_TYPE dt_x = cuda::solver_params.cfl * cuda::geometry.dx
					/ (FABS(U)+SQRT(cuda::physical_params.g * H));
				NUMERIC_TYPE dt_y = cuda::solver_params.cfl * cuda::geometry.dy
					/ (FABS(V)+SQRT(cuda::physical_params.g * H));

				dt[j*cuda::pitch + i] = FMIN(dt_x, dt_y);
			}
			else
			{
				dt[j*cuda::pitch + i] = cuda::solver_params.max_dt;
			}
		}
	}
}

__global__
void update_uniform_rain_func
(
	Flow U,
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE  rain_rate,
	const int* cell_mask
)
{
	int global_i = blockIdx.x * blockDim.x + threadIdx.x;
	int global_j = blockIdx.y * blockDim.y + threadIdx.y;

	for (int j = global_j; j < cuda::geometry.ysz+2; j += blockDim.y * gridDim.y)
	{
		for (int i = global_i; i < cuda::pitch; i += blockDim.x * gridDim.x)
		{
			if (cell_mask != nullptr)
			{
				if (i < 1 || i > cuda::geometry.xsz || j < 1 || j > cuda::geometry.ysz) continue;
				const int cls = cell_mask[(j-1)*cuda::geometry.xsz + (i-1)];
				if (cls == 0 || cls == 2) continue;
			}
			NUMERIC_TYPE cell_rain;

			cell_rain = rain_rate * cuda::dt;

			NUMERIC_TYPE Z = DEM[j * cuda::pitch + i];
			if (FABS(Z - cuda::solver_params.nodata_elevation) < C(1e-6) /* || Z < C(0.0) */) {

			}
			else {

				NUMERIC_TYPE& Hval = U.H[j * cuda::pitch + i];
				Hval += cell_rain;
			}
		}
	}
}

__global__ void update_ghost_cells
(
	Flow U,
	NUMERIC_TYPE* Zstar_x,
	NUMERIC_TYPE* Zstar_y,
	const int* cell_mask
)
{
	for (int j=blockIdx.x*blockDim.x+threadIdx.x+1; j<cuda::geometry.ysz+1;
			j+=blockDim.x*gridDim.x)
	{
		{
			int i = 1;
			if (!active_cell(cell_mask, i, j))
			{
				U.H[j*cuda::pitch] = U.HU[j*cuda::pitch] = U.HV[j*cuda::pitch] = C(0.0);
			}
			else
			{
			NUMERIC_TYPE Zstar = Zstar_x[j*cuda::pitch + i-1];
			FlowVector U_inside = U[j*cuda::pitch + i];
			FlowVector U_outside = Boundary::outside_x(U_inside, U_inside,
					Boundary::index_w(i, j), 1, Zstar);
			U.H[j*cuda::pitch + i-1] = U_outside.H;
			U.HU[j*cuda::pitch + i-1] = U_outside.HU;
			U.HV[j*cuda::pitch + i-1] = U_outside.HV;
			}
		}
		{
			int i = cuda::geometry.xsz;
			if (!active_cell(cell_mask, i, j))
			{
				const int g = j*cuda::pitch + i+1;
				U.H[g] = U.HU[g] = U.HV[g] = C(0.0);
			}
			else
			{
			NUMERIC_TYPE Zstar = Zstar_x[j*cuda::pitch + i];
			FlowVector U_inside = U[j*cuda::pitch + i];
			FlowVector U_outside = Boundary::outside_x(U_inside, U_inside,
					Boundary::index_e(i, j), -1, Zstar);
			U.H[j*cuda::pitch + i+1] = U_outside.H;
			U.HU[j*cuda::pitch + i+1] = U_outside.HU;
			U.HV[j*cuda::pitch + i+1] = U_outside.HV;
			}
		}
	}

	for (int i=blockIdx.x*blockDim.x+threadIdx.x+1; i<cuda::geometry.xsz+1;
			i+=blockDim.x*gridDim.x)
	{
		{
			int j = 1;
			if (!active_cell(cell_mask, i, j))
			{
				const int g = (j-1)*cuda::pitch + i;
				U.H[g] = U.HU[g] = U.HV[g] = C(0.0);
			}
			else
			{
			NUMERIC_TYPE Zstar = Zstar_y[(j-1)*cuda::pitch + i];
			FlowVector U_inside = U[j*cuda::pitch + i];
			FlowVector U_outside = Boundary::outside_y(U_inside, U_inside,
					Boundary::index_n(i, j), -1, Zstar);
			U.H[(j-1)*cuda::pitch + i] = U_outside.H;
			U.HU[(j-1)*cuda::pitch + i] = U_outside.HU;
			U.HV[(j-1)*cuda::pitch + i] = U_outside.HV;
			}
		}
		{
			int j = cuda::geometry.ysz;
			if (!active_cell(cell_mask, i, j))
			{
				const int g = (j+1)*cuda::pitch + i;
				U.H[g] = U.HU[g] = U.HV[g] = C(0.0);
			}
			else
			{
			NUMERIC_TYPE Zstar = Zstar_y[j*cuda::pitch + i];
			FlowVector U_inside = U[j*cuda::pitch + i];
			FlowVector U_outside = Boundary::outside_y(U_inside, U_inside,
					Boundary::index_s(i, j), 1, Zstar);
			U.H[(j+1)*cuda::pitch + i] = U_outside.H;
			U.HU[(j+1)*cuda::pitch + i] = U_outside.HU;
			U.HV[(j+1)*cuda::pitch + i] = U_outside.HV;
			}
		}
	}
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
apply_friction
(
	Flow U,
	NUMERIC_TYPE* manning,
	const int* cell_mask
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;

	for (int j=global_j+1; j<cuda::geometry.ysz+1; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i+1; i<cuda::geometry.xsz+1; i+=blockDim.x*gridDim.x)
		{
			if (!active_cell(cell_mask, i, j)) continue;
			NUMERIC_TYPE H = U.H[j*cuda::pitch + i];
			NUMERIC_TYPE& HU = U.HU[j*cuda::pitch + i];
			NUMERIC_TYPE& HV = U.HV[j*cuda::pitch + i];

			if (H <= cuda::solver_params.DepthThresh) {
				HU = C(0.0);
				HV = C(0.0);
				continue;
			}

			NUMERIC_TYPE U = HU/H;
			NUMERIC_TYPE V = HV/H;
			if (FABS(U) <= cuda::solver_params.SpeedThresh
					&& FABS(V) <= cuda::solver_params.SpeedThresh)
			{
				HU = C(0.0);
				HV = C(0.0);
				continue;
			}

            NUMERIC_TYPE n = (manning == nullptr)
                ? cuda::physical_params.manning
				: manning[j*cuda::pitch + i];

			NUMERIC_TYPE Cf = cuda::physical_params.g * n * n /
				POW(H, C(1.0)/C(3.0));
			NUMERIC_TYPE speed = SQRT(U*U+V*V);

			NUMERIC_TYPE Sf_x = -Cf*U*speed;
			NUMERIC_TYPE Sf_y = -Cf*V*speed;
			NUMERIC_TYPE D_x = C(1.0) + cuda::dt*Cf / H
				* (C(2.0)*U*U+V*V)/speed;
			NUMERIC_TYPE D_y = C(1.0) + cuda::dt*Cf / H
				* (U*U+C(2.0)*V*V)/speed;

			HU += cuda::dt*Sf_x/D_x;
			HV += cuda::dt*Sf_y/D_y;
		}
	}
}

__device__ NUMERIC_TYPE bed_source_x
(
 	NUMERIC_TYPE Zstar_w,
	NUMERIC_TYPE Zstar_e,
	NUMERIC_TYPE Hstar_w,
	NUMERIC_TYPE Hstar_e,
	NUMERIC_TYPE ETA
)
{
	NUMERIC_TYPE Zdagger_w = Zstar_w - FMAX(C(0.0), -(ETA - Zstar_w));
	NUMERIC_TYPE Zdagger_e = Zstar_e - FMAX(C(0.0), -(ETA - Zstar_e));
	return -cuda::physical_params.g * C(0.5) * (Hstar_w + Hstar_e)
		* (Zdagger_e - Zdagger_w)/cuda::geometry.dx;
}

__device__ NUMERIC_TYPE bed_source_y
(
 	NUMERIC_TYPE Zstar_s,
	NUMERIC_TYPE Zstar_n,
	NUMERIC_TYPE Hstar_s,
	NUMERIC_TYPE Hstar_n,
	NUMERIC_TYPE ETA
)
{
	NUMERIC_TYPE Zdagger_s = Zstar_s - FMAX(C(0.0), -(ETA - Zstar_s));
	NUMERIC_TYPE Zdagger_n = Zstar_n - FMAX(C(0.0), -(ETA - Zstar_n));
	return -cuda::physical_params.g * C(0.5) * (Hstar_s + Hstar_n)
		* (Zdagger_n - Zdagger_s)/cuda::geometry.dy;
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
update_flow_variables_x
(
	Flow Uold,
	Flow U,
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE* Zstar_x,
	MassStats* mass_stats,
	NUMERIC_TYPE* negative_depth_volume,
	const int* cell_mask
)
{
	__shared__ FlowVector F[CUDA_BLOCK_SIZE_Y][CUDA_BLOCK_SIZE_X];
	__shared__ NUMERIC_TYPE Hstar[CUDA_BLOCK_SIZE_Y][CUDA_BLOCK_SIZE_X];
	int global_i = blockIdx.x*(blockDim.x-1) + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;
	int max_i = ((cuda::geometry.xsz+blockDim.x-2)/(blockDim.x-1))*(blockDim.x-1);
	int max_j = ((cuda::geometry.ysz+blockDim.y-1)/blockDim.y)*blockDim.y;
	for (int j=global_j+1; j<=max_j; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i; i<=max_i; i+=(blockDim.x-1)*gridDim.x)
		{
			if (threadIdx.x == 0 && i == max_i) continue;
			FlowVector U_neg;
			FlowVector Ustar_neg;
			FlowVector F_e;
			NUMERIC_TYPE Z_neg;
			NUMERIC_TYPE Zstar_e;
			if (i <= cuda::geometry.xsz && j <= cuda::geometry.ysz)
			{
				Zstar_e = Zstar_x[j*cuda::pitch + i];
				const bool neg_active = i > 0 && active_cell(cell_mask, i, j);
				const bool pos_active = i < cuda::geometry.xsz && active_cell(cell_mask, i+1, j);
				const bool outer_active = (i == 0 && pos_active) ||
					(i == cuda::geometry.xsz && neg_active);
				const bool internal_active = i > 0 && i < cuda::geometry.xsz &&
					neg_active && pos_active;

				if (internal_active || outer_active)
				{
					Z_neg = DEM[j*cuda::pitch + i];
					U_neg = Uold[j*cuda::pitch + i];
					Ustar_neg = U_neg.star(Z_neg, Zstar_e);
					NUMERIC_TYPE Z_pos = DEM[j*cuda::pitch + i+1];
					FlowVector U_pos = Uold[j*cuda::pitch + i+1];
					FlowVector Ustar_pos = U_pos.star(Z_pos, Zstar_e);
					if (i == 0)
						Ustar_pos = Boundary::inside_x(Ustar_neg, Ustar_pos, Ustar_pos, Boundary::index_w(i, j));
					else if (i == cuda::geometry.xsz)
						Ustar_neg = Boundary::inside_x(Ustar_pos, Ustar_neg, Ustar_neg, Boundary::index_e(i, j));
					Hstar[threadIdx.y][threadIdx.x] = Ustar_pos.H;
					F_e = HLL::x(Ustar_neg, Ustar_pos);
				}
				else
				{
					F_e = { C(0.0), C(0.0), C(0.0) };
					Hstar[threadIdx.y][threadIdx.x] = C(0.0);
					if (neg_active)
					{
						Z_neg = DEM[j*cuda::pitch + i];
						U_neg = Uold[j*cuda::pitch + i];
						Ustar_neg = U_neg.star(Z_neg, Zstar_e);
					}
					if (pos_active)
					{
						const NUMERIC_TYPE Z_pos = DEM[j*cuda::pitch + i+1];
						const FlowVector U_pos = Uold[j*cuda::pitch + i+1];
						Hstar[threadIdx.y][threadIdx.x] = U_pos.star(Z_pos, Zstar_e).H;
					}
				}
				F[threadIdx.y][threadIdx.x] = F_e;
			}
			__syncthreads();
			// Keep the HLL boundary flux from the extrapolated ghost state.
			// Copying an interior-face flux here breaks the bed-source balance.
			if (i <= cuda::geometry.xsz && j <= cuda::geometry.ysz)
			{
				F_e = F[threadIdx.y][threadIdx.x];
				update_mass_stats_x(mass_stats, F_e.H, i, j);
			}
			if (threadIdx.x == 0) continue;
			if (i <= cuda::geometry.xsz && j <= cuda::geometry.ysz)
			{
				if (!active_cell(cell_mask, i, j))
				{
					U.H[j*cuda::pitch + i] = C(0.0);
					U.HU[j*cuda::pitch + i] = C(0.0);
					U.HV[j*cuda::pitch + i] = C(0.0);
					continue;
				}
				FlowVector& F_w = F[threadIdx.y][threadIdx.x-1];
				FlowVector U0 = Uold[j*cuda::pitch + i];
				NUMERIC_TYPE& H = U.H[j*cuda::pitch + i];
				NUMERIC_TYPE& HU = U.HU[j*cuda::pitch + i];
				NUMERIC_TYPE& HV = U.HV[j*cuda::pitch + i];
				NUMERIC_TYPE Zstar_w = Zstar_x[j*cuda::pitch + i-1];
				NUMERIC_TYPE Hstar_w = Hstar[threadIdx.y][threadIdx.x-1];
				NUMERIC_TYPE ETA = U_neg.H + Z_neg;
				H = U0.H - cuda::dt * (F_e.H - F_w.H)/cuda::geometry.dx;
				HU = U0.HU - cuda::dt * ((F_e.HU - F_w.HU)/cuda::geometry.dx
					- bed_source_x(Zstar_w, Zstar_e, Hstar_w, Ustar_neg.H, ETA));
				HV = U0.HV - cuda::dt * (F_e.HV - F_w.HV)/cuda::geometry.dx;
			}
		}
	}
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
update_flow_variables_y
(
	Flow Uold,
	Flow Uint,
	Flow U,
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE* Zstar_y,
	MassStats* mass_stats,
	NUMERIC_TYPE* negative_depth_volume,
	const int* cell_mask
)
{
	__shared__ FlowVector F[CUDA_BLOCK_SIZE_Y][CUDA_BLOCK_SIZE_X];
	__shared__ NUMERIC_TYPE Hstar[CUDA_BLOCK_SIZE_Y][CUDA_BLOCK_SIZE_X];
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*(blockDim.y-1) + threadIdx.y;
	int max_i = ((cuda::geometry.xsz+blockDim.x-1)/blockDim.x)*blockDim.x;
	int max_j = ((cuda::geometry.ysz+blockDim.y-2)/(blockDim.y-1))*(blockDim.y-1);
	for (int j=global_j; j<=max_j; j+=(blockDim.y-1)*gridDim.y)
	{
		if (threadIdx.y == 0 && j == max_j) continue;
		for (int i=global_i+1; i<=max_i; i+=blockDim.x*gridDim.x)
		{
			FlowVector U_pos;
			FlowVector Ustar_pos;
			FlowVector F_s;
			NUMERIC_TYPE Z_pos;
			NUMERIC_TYPE Zstar_s;
			if (i <= cuda::geometry.xsz && j <= cuda::geometry.ysz)
			{
				Zstar_s = Zstar_y[j*cuda::pitch + i];
				const bool pos_active = j > 0 && active_cell(cell_mask, i, j);
				const bool neg_active = j < cuda::geometry.ysz && active_cell(cell_mask, i, j+1);
				const bool outer_active = (j == 0 && neg_active) ||
					(j == cuda::geometry.ysz && pos_active);
				const bool internal_active = j > 0 && j < cuda::geometry.ysz &&
					neg_active && pos_active;

				if (internal_active || outer_active)
				{
					NUMERIC_TYPE Z_neg = DEM[(j+1)*cuda::pitch + i];
					FlowVector U_neg = Uold[(j+1)*cuda::pitch + i];
					FlowVector Ustar_neg = U_neg.star(Z_neg, Zstar_s);
					Z_pos = DEM[j*cuda::pitch + i];
					U_pos = Uold[j*cuda::pitch + i];
					Ustar_pos = U_pos.star(Z_pos, Zstar_s);
					if (j == 0)
						Ustar_neg = Boundary::inside_y(Ustar_pos, Ustar_neg, Ustar_neg, Boundary::index_n(i, j));
					else if (j == cuda::geometry.ysz)
						Ustar_pos = Boundary::inside_y(Ustar_neg, Ustar_pos, Ustar_pos, Boundary::index_s(i, j));
					Hstar[threadIdx.y][threadIdx.x] = Ustar_neg.H;
					F_s = HLL::y(Ustar_neg, Ustar_pos);
				}
				else
				{
					F_s = { C(0.0), C(0.0), C(0.0) };
					Hstar[threadIdx.y][threadIdx.x] = C(0.0);
					if (pos_active)
					{
						Z_pos = DEM[j*cuda::pitch + i];
						U_pos = Uold[j*cuda::pitch + i];
						Ustar_pos = U_pos.star(Z_pos, Zstar_s);
					}
					if (neg_active)
					{
						const NUMERIC_TYPE Z_neg = DEM[(j+1)*cuda::pitch + i];
						const FlowVector U_neg = Uold[(j+1)*cuda::pitch + i];
						Hstar[threadIdx.y][threadIdx.x] = U_neg.star(Z_neg, Zstar_s).H;
					}
				}
				F[threadIdx.y][threadIdx.x] = F_s;
			}
			__syncthreads();
			// Keep the HLL boundary flux from the extrapolated ghost state.
			// Copying an interior-face flux here breaks the bed-source balance.
			if (i <= cuda::geometry.xsz && j <= cuda::geometry.ysz)
			{
				F_s = F[threadIdx.y][threadIdx.x];
				update_mass_stats_y(mass_stats, F_s.H, i, j);
			}
			if (threadIdx.y == 0) continue;
			if (i <= cuda::geometry.xsz && j <= cuda::geometry.ysz)
			{
				if (!active_cell(cell_mask, i, j))
				{
					U.H[j*cuda::pitch + i] = C(0.0);
					U.HU[j*cuda::pitch + i] = C(0.0);
					U.HV[j*cuda::pitch + i] = C(0.0);
					continue;
				}
				FlowVector& F_n = F[threadIdx.y-1][threadIdx.x];
				FlowVector U0 = Uint[j*cuda::pitch + i];
				NUMERIC_TYPE& H = U.H[j*cuda::pitch + i];
				NUMERIC_TYPE& HU = U.HU[j*cuda::pitch + i];
				NUMERIC_TYPE& HV = U.HV[j*cuda::pitch + i];
				NUMERIC_TYPE Zstar_n = Zstar_y[(j-1)*cuda::pitch + i];
				NUMERIC_TYPE Hstar_n = Hstar[threadIdx.y-1][threadIdx.x];
				NUMERIC_TYPE ETA = U_pos.H + Z_pos;
				H = U0.H - cuda::dt * (F_n.H - F_s.H)/cuda::geometry.dy;
				HU = U0.HU - cuda::dt * (F_n.HU - F_s.HU)/cuda::geometry.dy;
				HV = U0.HV - cuda::dt * ((F_n.HV - F_s.HV)/cuda::geometry.dy
					- bed_source_y(Zstar_s, Zstar_n, Ustar_pos.H, Hstar_n, ETA));
			}
		}
	}
}

}
}
}

lis::cuda::fv1::Solver::Solver
(
	Flow& U,
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE* Zstar_x,
	NUMERIC_TYPE* Zstar_y,
	NUMERIC_TYPE* manning,
	Geometry& geometry,
	PhysicalParams& physical_params,
	dim3 grid_size
)
:
Uold(U1),
U(U2),
DEM(DEM),
Zstar_x(Zstar_x),
Zstar_y(Zstar_y),
manning(manning),
negative_depth_volume(nullptr),
cell_mask(nullptr),
sparse_faces(nullptr),
sparse_fluxes(nullptr),
sparse_face_count(0),
sparse_flux_capacity(0),
grid_size(grid_size)
{
	Flow::allocate_device(U1, geometry);
	Flow::allocate_device(U2, geometry);
	Flow::allocate_device(Ux, geometry);
	Flow::copy(U1, U, geometry);
	friction = (physical_params.manning > C(0.0)) || (manning != nullptr);
}

void lis::cuda::fv1::Solver::zero_ghost_cells()
{
	zero_ghost_cells_north_south<<<1, CUDA_BLOCK_SIZE>>>(U);
	zero_ghost_cells_east_west<<<1, CUDA_BLOCK_SIZE>>>(U);
}

void lis::cuda::fv1::Solver::update_ghost_cells(cudaStream_t stream)
{
	lis::cuda::fv1::update_ghost_cells<<<64, CUDA_BLOCK_SIZE, 0, stream>>>(
			Uold, Zstar_x, Zstar_y, cell_mask);
}

void lis::cuda::fv1::Solver::clamp_negative_depths(cudaStream_t stream)
{
	clamp_negative_depths_kernel<<<grid_size, cuda::block_size, 0, stream>>>(
			Uold, negative_depth_volume, cell_mask);
}

void lis::cuda::fv1::Solver::set_sparse_faces(const SparseFace* faces, int count)
{
	sparse_faces = faces;
	sparse_face_count = count > 0 ? count : 0;
	if (sparse_face_count > sparse_flux_capacity)
	{
		if (sparse_fluxes != nullptr) cuda::free_device(sparse_fluxes);
		sparse_fluxes = static_cast<SparseFaceFlux*>(cuda::malloc_device(
				static_cast<size_t>(sparse_face_count) * sizeof(SparseFaceFlux)));
		sparse_flux_capacity = sparse_face_count;
	}
}

void lis::cuda::fv1::Solver::set_cell_mask(const int* mask)
{
	cell_mask = mask;
	if (cell_mask == nullptr) return;
	apply_mask_to_zstar_x<<<grid_size, cuda::block_size>>>(DEM, Zstar_x, cell_mask);
	apply_mask_to_zstar_y<<<grid_size, cuda::block_size>>>(DEM, Zstar_y, cell_mask);
}

lis::cuda::fv1::Flow& lis::cuda::fv1::Solver::update_flow_variables
(
	MassStats* mass_stats,
	cudaStream_t stream
)
{
	if (friction)
	{
		apply_friction<<<grid_size, cuda::block_size, 0, stream>>>(Uold, manning, cell_mask);
		update_ghost_cells(stream);
	}

	if (sparse_face_count > 0 && sparse_faces != nullptr)
	{
		const int blocks = (sparse_face_count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
		prepare_sparse_face_fluxes<<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(
				Uold, DEM, Zstar_x, Zstar_y, sparse_faces, sparse_fluxes, sparse_face_count);
	}

	update_flow_variables_x<<<grid_size, cuda::block_size, 0, stream>>>
			(Uold, Ux, DEM, Zstar_x, mass_stats, negative_depth_volume, cell_mask);
	update_flow_variables_y<<<grid_size, cuda::block_size, 0, stream>>>
			(Uold, Ux, U, DEM, Zstar_y, mass_stats, negative_depth_volume, cell_mask);

	if (sparse_face_count > 0 && sparse_faces != nullptr)
	{
		const int blocks = (sparse_face_count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
		apply_sparse_face_fluxes<<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(
				U, sparse_faces, sparse_fluxes, sparse_face_count, mass_stats);
	}
	std::swap(Uold, U);
	return Uold;
}

lis::cuda::fv1::Flow& lis::cuda::fv1::Solver::d_U()
{
	return Uold;
}

void lis::cuda::fv1::Solver::update_dt_per_element
(
	NUMERIC_TYPE* dt_field
) const
{
	lis::cuda::fv1::update_dt_per_element<<<grid_size, cuda::block_size>>>(
			dt_field, Uold, cell_mask);
}

void lis::cuda::fv1::Solver::swap_state()
{
	std::swap(Uold, U);
}

void lis::cuda::fv1::Solver::FloodplainQ()
{
	return;
}

void lis::cuda::fv1::Solver::zero_thin_depth_slopes()
{
	return;
}

void lis::cuda::fv1::Solver::updateMaxFieldACC
(
	NUMERIC_TYPE t
)
{
	return;
}

void lis::cuda::fv1::Solver::update_uniform_rain
(
	NUMERIC_TYPE rain_rate,
	cudaStream_t stream
)
{
	update_uniform_rain_func<<<grid_size, cuda::block_size, 0, stream>>>(
		Uold, DEM, rain_rate, cell_mask);
}

lis::cuda::fv1::Solver::~Solver()
{
	Flow::free_device(U1);
	Flow::free_device(U2);
	Flow::free_device(Ux);
	if (sparse_fluxes != nullptr) cuda::free_device(sparse_fluxes);
}
