#ifndef SJB_CUDA_GPU_HELLS_CUH
#define SJB_CUDA_GPU_HELLS_CUH

#include <atomic>
#include <sjb/windowing/lazy/executor/BaseLazyExecutor.hpp>

// BITMAP CONFIG
using BitmapType = uint32_t;
const int BitmapCapacity = 8 * sizeof(BitmapType);

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            class GPU_HELLS : public BaseLazyExecutor {

            public:
                explicit GPU_HELLS(bool writeJoinResultToSink, uint64_t batchSize);

                ~GPU_HELLS();

                ExecutionStatistic execute(Tuple *leftRingBuffer, Tuple *rightRingBuffer,
                                           std::vector<uint64_t> leftIndexesToCopy,
                                           std::vector<uint64_t> rightIndexesToCopy,
                                           Sink &sink) final;

                void launchKernel(const Tuple *leftTuples, const Tuple *rightTuples,
                                  uint64_t nLeftTuples,
                                  uint64_t nRightTuples, BitmapType *bitmaps);

            private:
                uint64_t batchSize;
            };
        }
    }
}

#endif //SJB_CUDA_GPU_HELLS_CUH