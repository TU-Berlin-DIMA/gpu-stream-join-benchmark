#ifndef STREAMJOINBENCHMARK_EAGER_GPU_HJ_PK_CUH
#define STREAMJOINBENCHMARK_EAGER_GPU_HJ_PK_CUH

#include <sjb/windowing/eager/executor/BaseEagerExecutor.hpp>
#include <atomic>
#include <tbb/concurrent_queue.h>

#define CUDASTREAMCOUNT 1

namespace Windowing {
    namespace Eager {
        namespace Executor {
            class GPU_HJ_PK : public BaseEagerExecutor {
            public:
                GPU_HJ_PK(uint64_t numDistinctKeys, uint64_t windowSize, uint64_t batchSize);

                ~GPU_HJ_PK();

                uint64_t execute(Tuple *tupleBuffer, bool isLeftSide) final;

                void clearStates() final;

                void onEndOfStream() override;

            private:
                uint32_t nPersistentThreads = 1024;

                // Signals for the left build kernel
                int8_t *leftBuildWorkSignal;
                int8_t *leftBuildKeepAliveSignal;
                cudaStream_t leftBuildStream;

                // Signals for the left probe kernel
                int8_t *leftProbeWorkSignal;
                int8_t *leftProbeKeepAliveSignal;
                cudaStream_t leftProbeStream;

                // Signals for the right build kernel
                int8_t *rightBuildWorkSignal;
                int8_t *rightBuildKeepAliveSignal;
                cudaStream_t rightBuildStream;

                // Signals for the right probe kernel
                int8_t *rightProbeWorkSignal;
                int8_t *rightProbeKeepAliveSignal;
                cudaStream_t rightProbeStream;

                uint64_t numDistinctKeys;
                uint64_t windowSize;
                uint64_t batchSize;

                Tuple *leftHashTable;
                Tuple *rightHashTable;

                uint32_t *leftHashTableOccupation;
                uint32_t *rightHashTableOccupation;

                Tuple *leftWorkBuffer;
                Tuple *rightWorkBuffer;

                uint32_t *leftGlobalCounter;
                uint32_t *rightGlobalCounter;

                enum KernelIdentifier : uint8_t {
                    LEFT_BUILD_KERNEL = 0,
                    LEFT_PROBE_KERNEL = 1,
                    RIGHT_BUILD_KERNEL = 2,
                    RIGHT_PROBE_KERNEL = 3,
                };

                static void triggerKernelExecution(volatile int8_t *signals, uint32_t nThreadsToUse) ;

                void instantiateKernel(volatile int8_t *workSignal, volatile int8_t *keepAliveSignal,
                                       uint8_t kernelIdentifier, cudaStream_t stream, Tuple *workBuffer) const;

            };
        }
    }
}
#endif //STREAMJOINBENCHMARK_EAGER_GPU_HJ_PK_CUH
