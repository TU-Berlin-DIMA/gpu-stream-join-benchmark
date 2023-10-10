#ifndef STREAMJOINBENCHMARK_GPU_NLJ_BM_CUH
#define STREAMJOINBENCHMARK_GPU_NLJ_BM_CUH

#include <sjb/windowing/eager/executor/BaseEagerExecutor.hpp>
#include <atomic>
#include <sjb/utils/Tuple.hpp>
#include <sjb/windowing/lazy/executor/GPU/GPU_NLJ.cuh>

#define SetBit(data, y)    (data |= (1 << y))   /* Set Data.Y to 1      */

namespace Windowing {
    namespace Eager {
        namespace Executor {
            class GPU_NLJ_BM : public BaseEagerExecutor {
            public:
                GPU_NLJ_BM(uint64_t batchSize, uint64_t windowSize);

                ~GPU_NLJ_BM();

                uint64_t execute(Tuple *tupleBuffer, bool isLeftSide, uint64_t windowTupleCount,Sink &sink) final;

                void clearStates() final;

                void onEndOfStream() override;

            private:
                uint64_t batchSize;
                uint64_t windowSize;

                Tuple *d_leftTuples{};
                Tuple *d_rightTuples{};

                std::atomic<uint64_t> leftOccupation{0};
                std::atomic<uint64_t> rightOccupation{0};

                unsigned long long *d_counter;
                unsigned long long *h_counter;

                void launchNestedLoopBitmapKernel(const Tuple *incoming, const Tuple *existing,
                                                  const uint64_t nIncoming, uint64_t nExisting,
                                                  cudaStream_t cudastream,Sink &sink);
            };
        }
    }
}

#endif //STREAMJOINBENCHMARK_GPU_NLJ_BM_CUH
