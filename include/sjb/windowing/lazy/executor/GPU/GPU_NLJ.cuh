#ifndef SJB_CUDA_GPU_NLJ_CUH
#define SJB_CUDA_GPU_NLJ_CUH

#include <atomic>
#include <cuda_runtime.h>

#include <sjb/windowing/lazy/executor/BaseLazyExecutor.hpp>
#include <sjb/utils/ResultTuple.hpp>

// BITMAP CONFIG
using BitmapType = uint32_t;
const int BitmapCapacity = 8 * sizeof(BitmapType);

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            class GPU_NLJ : public BaseLazyExecutor {

            public:
                GPU_NLJ(uint64_t batchSize, uint64_t maxTupleInWindow);

                ~GPU_NLJ();

                ExecutionStatistic execute(Tuple *leftRingBuffer, Tuple *rightRingBuffer,
                                           std::vector<uint64_t> leftIndexesToCopy,
                                           std::vector<uint64_t> rightIndexesToCopy,
                                           Sink &sink) override = 0;

            protected:
                static void copyTuplesToLocalStore(Tuple *localStore, Tuple *ringBuffer, std::vector<uint64_t> indexes,
                                                           uint64_t batchSize, cudaStream_t cudaStream);

                static void launchNestedLoopCountKernel(const Tuple *leftTuples, const Tuple *rightTuples, uint64_t nLeftTuples,
                                                        uint64_t nRightTuples, unsigned long long *counter, cudaStream_t cudaStream);

                static void launchNestedLoopJoinKernel(const Tuple *leftTuples, const Tuple *rightTuples,
                                                       const uint64_t nLeftTuples, uint64_t nRightTuples,
                                                       unsigned long long int *counter,  ResultTuple *d_resultTuple,
                                                       cudaStream_t cudaStream);

                static void launchNestedLoopBitmapKernel(const Tuple *leftTuples, const Tuple *rightTuples, const uint64_t nLeftTuples,
                                                         uint64_t nRightTuples, BitmapType *bitmaps, cudaStream_t cudastream);

                uint64_t batchSize;
                uint64_t maxTupleInWindow;

                cudaStream_t joinerStream;
                cudaStream_t leftStream;
                cudaStream_t rightStream;

                unsigned long long *h_resultCount;
                unsigned long long *d_resultCount;

                Tuple *d_leftTuples;
                Tuple *d_rightTuples;
            };
        }
    }
}

#endif //SJB_CUDA_GPU_NLJ_CUH