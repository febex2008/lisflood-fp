#pragma once
#include "../lisflood.h"
namespace lis { namespace cuda {
extern __constant__ const NUMERIC_TYPE* storage_depth_grid;
__device__ inline NUMERIC_TYPE cell_storage_depth(int g) {
    return storage_depth_grid ? storage_depth_grid[g] : C(0.0);
}
__device__ inline NUMERIC_TYPE cell_depth_threshold(int g, NUMERIC_TYPE minimum) {
    return FMAX(minimum, cell_storage_depth(g));
}
}}
