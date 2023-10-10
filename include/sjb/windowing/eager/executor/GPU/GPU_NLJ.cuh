#ifndef STREAMJOINBENCHMARK_GPU_NLJ_CUH
#define STREAMJOINBENCHMARK_GPU_NLJ_CUH

#include <sjb/windowing/eager/executor/BaseEagerExecutor.hpp>
#include <atomic>
#include <sjb/utils/Tuple.hpp>

namespace Windowing {
    namespace Eager {
        namespace Executor {
            class GPU_NLJ : public BaseEagerExecutor {
            public:
                GPU_NLJ(uint64_t batchSize, uint64_t windowSize);

                ~GPU_NLJ();

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


                uint64_t
                launchKernel(Tuple *leftTuples, Tuple *rightTuples, uint64_t nLeftTuples, uint64_t nRightTuples,
                             cudaStream_t cudaStream);
            };
        }
    }
}

#endif //STREAMJOINBENCHMARK_GPU_NLJ_CUH
