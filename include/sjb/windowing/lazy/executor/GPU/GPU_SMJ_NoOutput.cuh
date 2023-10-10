#ifndef SJB_CUDA_GPU_SMJ_CUH_NO_OUTPUT
#define SJB_CUDA_GPU_SMJ_CUH_NO_OUTPUT

#include <sjb/windowing/lazy/executor/GPU/GPU_SMJ.cuh>

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            class GPU_SMJ_NoOutput : public GPU_SMJ {
            public:
                GPU_SMJ_NoOutput(uint64_t numDistinctKeys, uint64_t batchSize,
                        uint64_t maxTupleInWindow);

                ~GPU_SMJ_NoOutput();

                ExecutionStatistic execute(Tuple *leftRingBuffer, Tuple *rightRingBuffer,
                                           std::vector<uint64_t> leftIndexesToCopy,
                                           std::vector<uint64_t> rightIndexesToCopy,
                                           Sink &sink) final;
            };
        }
    }
}


#endif //SJB_CUDA_GPU_SMJ_CUH_NO_OUTPUT
