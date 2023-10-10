#ifndef STREAMJOINBENCHMARK_GPU_NLJ_NOOUTPUT_CUH
#define STREAMJOINBENCHMARK_GPU_NLJ_NOOUTPUT_CUH

#include <sjb/windowing/lazy/executor/GPU/GPU_NLJ.cuh>

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            class GPU_NLJ_NoOutput : public GPU_NLJ {
            public:
                GPU_NLJ_NoOutput(uint64_t batchSize, uint64_t maxTupleInWindow);

                ~GPU_NLJ_NoOutput();

                ExecutionStatistic execute(Tuple *leftRingBuffer, Tuple *rightRingBuffer,
                                           std::vector<uint64_t> leftIndexesToJoin,
                                           std::vector<uint64_t> rightIndexesToJoin,
                                           Sink &sink) final;

            };
        }
    }
}

#endif //STREAMJOINBENCHMARK_GPU_NLJ_NOOUTPUT_CUH
