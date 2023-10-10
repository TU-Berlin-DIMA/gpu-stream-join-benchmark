#ifndef SJB_CUDA_CUDAUTILS_CUH
#define SJB_CUDA_CUDAUTILS_CUH

#include <cooperative_groups.h>

class CudaUtils {
public:
    static __device__ int atomicAggInc(unsigned long long int *ctr) {
        auto g = cooperative_groups::coalesced_threads();
        int warp_res;
        if (g.thread_rank() == 0)
            warp_res = atomicAdd(ctr, g.size());
        return g.shfl(warp_res, 0) + g.thread_rank();
    }
};


#endif //SJB_CUDA_CUDAUTILS_CUH
