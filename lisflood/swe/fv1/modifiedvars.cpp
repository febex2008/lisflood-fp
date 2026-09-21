#include "modifiedvars.h"

void fv1::initialise_Zstar
(
	Pars *Parptr,
	Arrays *Arrptr,
	const int *cell_mask
)
{
	// Internal x faces. A masked neighbour is replaced by a face ghost with
	// the active cell bed elevation; fully masked faces are not used.
#pragma omp parallel for
	for (int j=0; j<Parptr->ysz; j++)
	{
		for(int i=1; i<Parptr->xsz; i++)
		{
			const size_t neg = static_cast<size_t>(j)*Parptr->xsz + i-1;
			const size_t pos = neg + 1;
			const bool neg_active = cell_mask == nullptr || cell_mask[neg] != 0;
			const bool pos_active = cell_mask == nullptr || cell_mask[pos] != 0;
			const NUMERIC_TYPE Z_neg = Arrptr->DEM[neg];
			const NUMERIC_TYPE Z_pos = Arrptr->DEM[pos];
			NUMERIC_TYPE& Zstar_x = Arrptr->Zstar_x[j*(Parptr->xsz+1) + i];

			if (neg_active && pos_active) Zstar_x = getmax(Z_neg, Z_pos);
			else if (neg_active) Zstar_x = Z_neg;
			else if (pos_active) Zstar_x = Z_pos;
			else Zstar_x = C(0.0);
		}
	}

	// Internal y faces.
#pragma omp parallel for
	for (int j=1; j<Parptr->ysz; j++)
	{
		for(int i=0; i<Parptr->xsz; i++)
		{
			const size_t neg = static_cast<size_t>(j)*Parptr->xsz + i;
			const size_t pos = static_cast<size_t>(j-1)*Parptr->xsz + i;
			const bool neg_active = cell_mask == nullptr || cell_mask[neg] != 0;
			const bool pos_active = cell_mask == nullptr || cell_mask[pos] != 0;
			const NUMERIC_TYPE Z_neg = Arrptr->DEM[neg];
			const NUMERIC_TYPE Z_pos = Arrptr->DEM[pos];
			NUMERIC_TYPE& Zstar_y = Arrptr->Zstar_y[j*(Parptr->xsz+1) + i];

			if (neg_active && pos_active) Zstar_y = getmax(Z_neg, Z_pos);
			else if (neg_active) Zstar_y = Z_neg;
			else if (pos_active) Zstar_y = Z_pos;
			else Zstar_y = C(0.0);
		}
	}

	// Rectangular outer faces remain ordinary ghost boundaries where the
	// adjacent cell is active.
#pragma omp parallel for
	for (int j=0; j<Parptr->ysz; j++)
	{
		const size_t kw = static_cast<size_t>(j)*Parptr->xsz;
		const size_t ke = kw + Parptr->xsz - 1;
		Arrptr->Zstar_x[j*(Parptr->xsz+1)] =
			(cell_mask == nullptr || cell_mask[kw] != 0) ? Arrptr->DEM[kw] : C(0.0);
		Arrptr->Zstar_x[j*(Parptr->xsz+1) + Parptr->xsz] =
			(cell_mask == nullptr || cell_mask[ke] != 0) ? Arrptr->DEM[ke] : C(0.0);
	}

#pragma omp parallel for
	for (int i=0; i<Parptr->xsz; i++)
	{
		const size_t kn = i;
		const size_t ks = static_cast<size_t>(Parptr->ysz-1)*Parptr->xsz + i;
		Arrptr->Zstar_y[i] =
			(cell_mask == nullptr || cell_mask[kn] != 0) ? Arrptr->DEM[kn] : C(0.0);
		Arrptr->Zstar_y[Parptr->ysz*(Parptr->xsz+1) + i] =
			(cell_mask == nullptr || cell_mask[ks] != 0) ? Arrptr->DEM[ks] : C(0.0);
	}
}

void fv1::update_Hstar
(
	Pars *Parptr,
	Arrays *Arrptr,
	const int *cell_mask
)
{
#pragma omp parallel for
	for (int j=0; j<Parptr->ysz; j++)
	{
		for(int i=0; i<Parptr->xsz; i++)
		{
			const size_t k = static_cast<size_t>(j)*Parptr->xsz + i;
			NUMERIC_TYPE& Hstar_neg_x = Arrptr->Hstar_neg_x[k];
			if (cell_mask != nullptr && cell_mask[k] == 0) { Hstar_neg_x = C(0.0); continue; }
			const NUMERIC_TYPE ETA = eta(Parptr, Arrptr, i, j);
			const NUMERIC_TYPE Zstar_x = Arrptr->Zstar_x[j*(Parptr->xsz+1) + i+1];
			Hstar_neg_x = getmax(C(0.0), ETA - Zstar_x);
		}
	}

#pragma omp parallel for
	for (int j=0; j<Parptr->ysz; j++)
	{
		for(int i=0; i<Parptr->xsz; i++)
		{
			const size_t k = static_cast<size_t>(j)*Parptr->xsz + i;
			NUMERIC_TYPE& Hstar_pos_x = Arrptr->Hstar_pos_x[k];
			if (cell_mask != nullptr && cell_mask[k] == 0) { Hstar_pos_x = C(0.0); continue; }
			const NUMERIC_TYPE ETA = eta(Parptr, Arrptr, i, j);
			const NUMERIC_TYPE Zstar_x = Arrptr->Zstar_x[j*(Parptr->xsz+1) + i];
			Hstar_pos_x = getmax(C(0.0), ETA - Zstar_x);
		}
	}

#pragma omp parallel for
	for (int j=0; j<Parptr->ysz; j++)
	{
		for(int i=0; i<Parptr->xsz; i++)
		{
			const size_t k = static_cast<size_t>(j)*Parptr->xsz + i;
			NUMERIC_TYPE& Hstar_neg_y = Arrptr->Hstar_neg_y[k];
			if (cell_mask != nullptr && cell_mask[k] == 0) { Hstar_neg_y = C(0.0); continue; }
			const NUMERIC_TYPE ETA = eta(Parptr, Arrptr, i, j);
			const NUMERIC_TYPE Zstar_y = Arrptr->Zstar_y[j*(Parptr->xsz+1) + i];
			Hstar_neg_y = getmax(C(0.0), ETA - Zstar_y);
		}
	}

#pragma omp parallel for
	for (int j=0; j<Parptr->ysz; j++)
	{
		for(int i=0; i<Parptr->xsz; i++)
		{
			const size_t k = static_cast<size_t>(j)*Parptr->xsz + i;
			NUMERIC_TYPE& Hstar_pos_y = Arrptr->Hstar_pos_y[k];
			if (cell_mask != nullptr && cell_mask[k] == 0) { Hstar_pos_y = C(0.0); continue; }
			const NUMERIC_TYPE ETA = eta(Parptr, Arrptr, i, j);
			const NUMERIC_TYPE Zstar_y = Arrptr->Zstar_y[(j+1)*(Parptr->xsz+1) + i];
			Hstar_pos_y = getmax(C(0.0), ETA - Zstar_y);
		}
	}
}

NUMERIC_TYPE fv1::HUstar_neg_x
(
	Pars *Parptr,
	Solver *Solverptr,
	Arrays *Arrptr,
	const int i,
	const int j
)
{
	return discharge_star(
			Arrptr->Hstar_neg_x, Arrptr->HU, Parptr, Solverptr, Arrptr, i, j);
}

NUMERIC_TYPE fv1::HUstar_pos_x
(
	Pars *Parptr,
	Solver *Solverptr,
	Arrays *Arrptr,
	const int i,
	const int j
)
{
	return discharge_star(
			Arrptr->Hstar_pos_x, Arrptr->HU, Parptr, Solverptr, Arrptr, i, j);
}

NUMERIC_TYPE fv1::HUstar_neg_y
(
	Pars *Parptr,
	Solver *Solverptr,
	Arrays *Arrptr,
	const int i,
	const int j
)
{
	return discharge_star(
			Arrptr->Hstar_neg_y, Arrptr->HU, Parptr, Solverptr, Arrptr, i, j);
}

NUMERIC_TYPE fv1::HUstar_pos_y
(
	Pars *Parptr,
	Solver *Solverptr,
	Arrays *Arrptr,
	const int i,
	const int j
)
{
	return discharge_star(
			Arrptr->Hstar_pos_y, Arrptr->HU, Parptr, Solverptr, Arrptr, i, j);
}

NUMERIC_TYPE fv1::HVstar_neg_x
(
	Pars *Parptr,
	Solver *Solverptr,
	Arrays *Arrptr,
	const int i,
	const int j
)
{
	return discharge_star(
			Arrptr->Hstar_neg_x, Arrptr->HV, Parptr, Solverptr, Arrptr, i, j);
}

NUMERIC_TYPE fv1::HVstar_pos_x
(
	Pars *Parptr,
	Solver *Solverptr,
	Arrays *Arrptr,
	const int i,
	const int j
)
{
	return discharge_star(
			Arrptr->Hstar_pos_x, Arrptr->HV, Parptr, Solverptr, Arrptr, i, j);
}

NUMERIC_TYPE fv1::HVstar_neg_y
(
	Pars *Parptr,
	Solver *Solverptr,
	Arrays *Arrptr,
	const int i,
	const int j
)
{
	return discharge_star(
			Arrptr->Hstar_neg_y, Arrptr->HV, Parptr, Solverptr, Arrptr, i, j);
}

NUMERIC_TYPE fv1::HVstar_pos_y
(
	Pars *Parptr,
	Solver *Solverptr,
	Arrays *Arrptr,
	const int i,
	const int j
)
{
	return discharge_star(
			Arrptr->Hstar_pos_y, Arrptr->HV, Parptr, Solverptr, Arrptr, i, j);
}

NUMERIC_TYPE fv1::discharge_star
(
	NUMERIC_TYPE *Hstar,
	NUMERIC_TYPE *discharge_component,
	Pars *Parptr,
	Solver *Solverptr,
	Arrays *Arrptr,
	const int i,
	const int j
)
{
	NUMERIC_TYPE U = speed(
			discharge_component, Parptr, Solverptr, Arrptr, i, j);
	NUMERIC_TYPE Hstar_value = Hstar[j*Parptr->xsz + i];

	return Hstar_value * U;
}

NUMERIC_TYPE fv1::Zdagger_neg_x
(
	Pars *Parptr,
	Arrays *Arrptr,
	const int i,
	const int j
)
{
	return Zdagger(Parptr, Arrptr, Arrptr->Zstar_x, i+1, j, i, j);
}

NUMERIC_TYPE fv1::Zdagger_pos_x
(
	Pars *Parptr,
	Arrays *Arrptr,
	const int i,
	const int j
)
{
	return Zdagger(Parptr, Arrptr, Arrptr->Zstar_x, i, j, i, j);
}

NUMERIC_TYPE fv1::Zdagger_neg_y
(
	Pars *Parptr,
	Arrays *Arrptr,
	const int i,
	const int j
)
{
	return Zdagger(Parptr, Arrptr, Arrptr->Zstar_y, i, j, i, j);
}

NUMERIC_TYPE fv1::Zdagger_pos_y
(
	Pars *Parptr,
	Arrays *Arrptr,
	const int i,
	const int j
)
{
	return Zdagger(Parptr, Arrptr, Arrptr->Zstar_y, i, j+1, i, j);
}

NUMERIC_TYPE fv1::Zdagger
(
	Pars *Parptr,
	Arrays *Arrptr,
	NUMERIC_TYPE *Zstar,
	const int Zstar_i,
	const int Zstar_j,
	const int ETA_i,
	const int ETA_j
)
{
	NUMERIC_TYPE Zstar_value = Zstar[Zstar_j*(Parptr->xsz+1) + Zstar_i];
	NUMERIC_TYPE ETA = eta(Parptr, Arrptr, ETA_i, ETA_j);

	return Zstar_value - getmax(C(0.0), -(ETA - Zstar_value));
}

NUMERIC_TYPE fv1::eta
(
	Pars *Parptr,
	Arrays *Arrptr,
	const int i,
	const int j
)
{
	NUMERIC_TYPE Z = Arrptr->DEM[j*Parptr->xsz + i];
	NUMERIC_TYPE H = Arrptr->H[j*Parptr->xsz + i];

	return Z + H;
}

NUMERIC_TYPE fv1::speed
(
	NUMERIC_TYPE *discharge_component,
	Pars *Parptr,
	Solver *Solverptr,
	Arrays *Arrptr,
	const int i,
	const int j
)
{
	NUMERIC_TYPE H = Arrptr->H[j*Parptr->xsz + i];

	if (H > Solverptr->DepthThresh)
	{
		NUMERIC_TYPE discharge = discharge_component[j*Parptr->xsz + i];
		return discharge / H;
	}
	else
	{
		return C(0.0);
	}
}
