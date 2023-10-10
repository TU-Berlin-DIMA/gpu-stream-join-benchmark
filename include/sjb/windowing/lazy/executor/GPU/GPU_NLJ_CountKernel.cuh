#ifndef STREAMJOINBENCHMARK_NLJ_COUNTKERNEL_CUH
#define STREAMJOINBENCHMARK_NLJ_COUNTKERNEL_CUH

#include <sjb/windowing/lazy/executor/GPU/GPU_NLJ.cuh>

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            class GPU_NLJ_CountKernel : public GPU_NLJ {
            public:
                GPU_NLJ_CountKernel(uint64_t batchSize, uint64_t maxTupleInWindow);

                ~GPU_NLJ_CountKernel();

                ExecutionStatistic execute(Tuple *leftRingBuffer, Tuple *rightRingBuffer,
                                           std::vector<uint64_t> leftIndexesToJoin,
                                           std::vector<uint64_t> rightIndexesToJoin,
                                           Sink &sink) final;
            };
        }
    }
}

#endif //STREAMJOINBENCHMARK_NLJ_COUNTKERNEL_CUH
