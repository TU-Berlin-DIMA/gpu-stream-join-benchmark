#ifndef STREAMJOINBENCHMARK_GPU_NLJ_ESTIMATEDSELECTIVITY_CUH
#define STREAMJOINBENCHMARK_GPU_NLJ_ESTIMATEDSELECTIVITY_CUH

#include <sjb/windowing/lazy/executor/GPU/GPU_NLJ.cuh>

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            class GPU_NLJ_EstimatedSelectivity : public GPU_NLJ {
            public:
                GPU_NLJ_EstimatedSelectivity(uint64_t batchSize, uint64_t maxTupleInWindow,double estimatedSelectivity);

                ~GPU_NLJ_EstimatedSelectivity();

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

#endif //STREAMJOINBENCHMARK_GPU_NLJ_ESTIMATEDSELECTIVITY_CUH
