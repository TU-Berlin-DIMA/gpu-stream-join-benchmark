#ifndef STREAMJOINBENCHMARK_EAGER_GPU_PK_EMPTY_CUH
#define STREAMJOINBENCHMARK_EAGER_GPU_PK_EMPTY_CUH

#include <sjb/windowing/eager/executor/BaseEagerExecutor.hpp>
#include <atomic>
#include <tbb/concurrent_queue.h>

#define CUDASTREAMCOUNT 1

namespace Windowing {
    namespace Eager {
        namespace Executor {
            /**
             * Empty class that serves as a starting point to implement persistent kernel approach
             */
            class GPU_PK_Empty : public BaseEagerExecutor {
            public:
                GPU_PK_Empty(uint64_t numDistinctKeys, uint64_t windowSize, uint64_t batchSize);

                ~GPU_PK_Empty();

                uint64_t execute(Tuple *tupleBuffer, bool isLeftSide) final;

                void clearStates() final;

                void onEndOfStream() override;

            private:
                uint32_t nPersistentThreads = 1;
                int8_t *signals;
                int8_t *keepAlive;

                uint64_t numDistinctKeys;
                uint64_t windowSize;
                uint64_t batchSize;

                cudaStream_t *cudaStreams;

                void triggerKernelExecution(volatile int8_t *signals, uint32_t nPersistentThreads);
            };
        }
    }
}
#endif //STREAMJOINBENCHMARK_EAGER_GPU_PK_EMPTY_CUH
