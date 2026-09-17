#include "cuda_hll.cuh"
#include "cuda_solver.cuh"

__device__ lis::cuda::FlowVector lis::cuda::HLL::y
(
	const FlowVector& u_neg,
	const FlowVector& u_pos
)
{
	FlowVector u_neg_rotated = { u_neg.H, u_neg.HV, -u_neg.HU };
	FlowVector u_pos_rotated = { u_pos.H, u_pos.HV, -u_pos.HU };

	FlowVector F_rotated = HLL::x(u_neg_rotated, u_pos_rotated);

	return { F_rotated.H, -F_rotated.HV, F_rotated.HU };
}

__device__ lis::cuda::FlowVector lis::cuda::HLL::x
(
	const FlowVector& u_neg,
	const FlowVector& u_pos
)
{
	if (u_neg.H <= cuda::solver_params.DepthThresh &&
			u_pos.H <= cuda::solver_params.DepthThresh)
	{
		return FlowVector();
	}

	NUMERIC_TYPE U_neg = C(0.0);
	NUMERIC_TYPE V_neg = C(0.0);
	if (u_neg.H > cuda::solver_params.DepthThresh)
	{
		const NUMERIC_TYPE inv_H_neg = C(1.0) / u_neg.H;
		U_neg = u_neg.HU * inv_H_neg;
		V_neg = u_neg.HV * inv_H_neg;
	}

	NUMERIC_TYPE U_pos = C(0.0);
	NUMERIC_TYPE V_pos = C(0.0);
	if (u_pos.H > cuda::solver_params.DepthThresh)
	{
		const NUMERIC_TYPE inv_H_pos = C(1.0) / u_pos.H;
		U_pos = u_pos.HU * inv_H_pos;
		V_pos = u_pos.HV * inv_H_pos;
	}

	const NUMERIC_TYPE A_neg = SQRT(cuda::physical_params.g * u_neg.H);
	const NUMERIC_TYPE A_pos = SQRT(cuda::physical_params.g * u_pos.H);

	const NUMERIC_TYPE q_star = C(0.5) * (A_neg + A_pos)
		+ C(0.25) * (U_neg - U_pos);
	const NUMERIC_TYPE U_star = C(0.5) * (U_neg + U_pos) + A_neg - A_pos;
	const NUMERIC_TYPE A_star = FABS(q_star);

	NUMERIC_TYPE S_neg;
	if (u_neg.H <= cuda::solver_params.DepthThresh)
	{
		S_neg = U_pos - C(2.0) * A_pos;
	}
	else
	{
		S_neg = FMIN(U_neg - A_neg, U_star - A_star);
	}

	NUMERIC_TYPE S_pos;
	if (u_pos.H <= cuda::solver_params.DepthThresh)
	{
		S_pos = U_neg + C(2.0) * A_neg;
	}
	else
	{
		S_pos = FMAX(U_pos + A_pos, U_star + A_star);
	}

	if (S_neg >= C(0.0))
	{
		return {
			u_neg.HU,
			U_neg * u_neg.HU + C(0.5) * cuda::physical_params.g * u_neg.H * u_neg.H,
			u_neg.HV * U_neg
		};
	}
	else if (S_pos < C(0.0))
	{
		return {
			u_pos.HU,
			U_pos * u_pos.HU + C(0.5) * cuda::physical_params.g * u_pos.H * u_pos.H,
			u_pos.HV * U_pos
		};
	}
	else
	{
		const FlowVector F_neg = {
			u_neg.HU,
			U_neg * u_neg.HU + C(0.5) * cuda::physical_params.g * u_neg.H * u_neg.H,
			u_neg.HV * U_neg
		};
		const FlowVector F_pos = {
			u_pos.HU,
			U_pos * u_pos.HU + C(0.5) * cuda::physical_params.g * u_pos.H * u_pos.H,
			u_pos.HV * U_pos
		};

		const NUMERIC_TYPE inv_span = C(1.0) / (S_pos - S_neg);
		const NUMERIC_TYPE cross = S_neg * S_pos;
		FlowVector F;
		F.H = (S_pos * F_neg.H - S_neg * F_pos.H
			+ cross * (u_pos.H - u_neg.H)) * inv_span;
		F.HU = (S_pos * F_neg.HU - S_neg * F_pos.HU
			+ cross * (u_pos.HU - u_neg.HU)) * inv_span;

		const NUMERIC_TYPE pos_term = U_pos - S_pos;
		const NUMERIC_TYPE neg_term = U_neg - S_neg;
		const NUMERIC_TYPE S_mid =
			(S_neg * u_pos.H * pos_term - S_pos * u_neg.H * neg_term)
			/
			(u_pos.H * pos_term - u_neg.H * neg_term);

		F.HV = F.H * (S_mid >= C(0.0) ? V_neg : V_pos);
		return F;
	}
}