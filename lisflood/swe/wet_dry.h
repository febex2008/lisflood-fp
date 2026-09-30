#pragma once
#include "../lisflood.h"

// Activity depends only on physical depth and the local storage threshold.
#ifdef __CUDACC__
__host__ __device__
#endif
inline bool momentum_wet(NUMERIC_TYPE h, NUMERIC_TYPE, NUMERIC_TYPE,
                         NUMERIC_TYPE dry_depth, NUMERIC_TYPE storage = C(0.0))
{
    return h > (storage > dry_depth ? storage : dry_depth);
}

inline NUMERIC_TYPE surface_storage_depth(const Solver* solver, size_t cell)
{
    return solver->storage_depth ? solver->storage_depth[cell] : C(0.0);
}
inline bool surface_momentum_wet(const Solver* solver, size_t cell,
                                 NUMERIC_TYPE h, NUMERIC_TYPE hu, NUMERIC_TYPE hv)
{
    return momentum_wet(h, hu, hv, solver->DepthThresh, surface_storage_depth(solver,cell));
}
