#ifndef SJB_CUDA_GPU_SMJ_CUH_ESTIMATED_SELECTIVITY
#define SJB_CUDA_GPU_SMJ_CUH_ESTIMATED_SELECTIVITY

#include <sjb/windowing/lazy/executor/GPU/GPU_SMJ.cuh>

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            class GPU_SMJ_EstimatedSelectivity : public GPU_SMJ {
            public:
                GPU_SMJ_EstimatedSelectivity(uint64_t numDistinctKeys, uint64_t batchSize,
                                             uint64_t maxTupleInWindow, double estimatedSelectivity);

                ~GPU_SMJ_EstimatedSelectivity();

                ExecutionStatistic execute(Tuple *leftRingBuffer, Tuple *rightRingBuffer,
                                           std::vector<uint64_t> leftIndexesToCopy,
                                           std::vector<uint64_t> rightIndexesToCopy,
                                           Sink &sink) final;

            private:
                double estimatedSelectivity;
            };
        }
    }
}


#endif //SJB_CUDA_GPU_SMJ_CUH
