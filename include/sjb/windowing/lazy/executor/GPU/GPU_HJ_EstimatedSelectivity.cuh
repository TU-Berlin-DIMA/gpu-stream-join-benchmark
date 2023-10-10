#ifndef SJB_CUDA_GPU_HJ_CUH_ESTIMATED_SELECTIVITY
#define SJB_CUDA_GPU_HJ_CUH_ESTIMATED_SELECTIVITY

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include <sjb/windowing/lazy/executor/GPU/GPU_HJ.cuh>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            class GPU_HJ_EstimatedSelectivity : public GPU_HJ {
            public:
                GPU_HJ_EstimatedSelectivity(double estimatedSelectivity,
                                            uint64_t numDistinctKeys, uint64_t batchSize, uint64_t maxTupleInWindow);

                ~GPU_HJ_EstimatedSelectivity();

                ExecutionStatistic execute(Tuple *leftRingBuffer, Tuple *rightRingBuffer,
                                           std::vector<uint64_t> leftIndexesToJoin,
                                           std::vector<uint64_t> rightIndexesToJoin,
                                           Sink &sink) final;

            private:
                double estimatedSelectivity;
            };
        }
    }
}
#endif //SJB_CUDA_GPU_HJ_CUH_ESTIMATED_SELECTIVITY
