#ifndef SJB_CUDA_GPU_HJ_CUH_NO_OUTPUT
#define SJB_CUDA_GPU_HJ_CUH_NO_OUTPUT

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
            class GPU_HJ_NoOutput : public GPU_HJ {
            public:
                GPU_HJ_NoOutput(uint64_t numDistinctKeys,
                                uint64_t batchSize,
                                uint64_t maxTupleInWindow);

                ~GPU_HJ_NoOutput();

                ExecutionStatistic execute(Tuple *leftRingBuffer, Tuple *rightRingBuffer,
                                           std::vector<uint64_t> leftIndexesToJoin,
                                           std::vector<uint64_t> rightIndexesToJoin,
                                           Sink& sink) final;
            };
        }
    }
}
#endif //SJB_CUDA_GPU_HJ_CUH_ATOMIC
