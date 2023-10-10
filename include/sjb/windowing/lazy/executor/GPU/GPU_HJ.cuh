#ifndef STREAMJOINBENCHMARK_GPU_HJ_CUH
#define STREAMJOINBENCHMARK_GPU_HJ_CUH

#include <sjb/windowing/lazy/executor/BaseLazyExecutor.hpp>
#include <sjb/utils/ResultTuple.hpp>
#include <cuda_runtime.h>


namespace Windowing {
    namespace Lazy {
        namespace Executor {
            class GPU_HJ : public BaseLazyExecutor {
            public:
                GPU_HJ(uint64_t numDistinctKeys, uint64_t batchSize, uint64_t maxTupleInWindow);

                ~GPU_HJ();

                ExecutionStatistic execute(Tuple *leftRingBuffer, Tuple *rightRingBuffer,
                                           std::vector<uint64_t> leftIndexesToCopy,
                                           std::vector<uint64_t> rightIndexesToCopy,
                                           Sink &sink) override = 0;

            protected:
                static void launchComputeHistogramKernel(Tuple *buildSide,
                                                         uint64_t nBuildSideTuples,
                                                         unsigned long long int *histogram,
                                                         uint64_t numDistinctKeys,
                                                         cudaStream_t cudaStream);

                static void launchProbeJoinKernel(Tuple *rightTuples, uint64_t nRightTuple, uint64_t nLeftTuple,
                                                  Tuple *hashTable, const unsigned long long int *histogram,
                                                  const uint64_t *prefixSum, unsigned long long int *resultCount,
                                                  uint64_t numDistinctKeys, ResultTuple *resultTuple,
                                                  cudaStream_t cudaStream);

                static void launchProbeCountKernel(Tuple *rightTuples, uint64_t nRightTuple, uint64_t nLeftTuple,
                                                   Tuple *hashTable,
                                                   const unsigned long long int *histogram, const uint64_t *prefixSum,
                                                   unsigned long long int *resultCount, uint64_t numDistinctKeys,
                                                   cudaStream_t cudaStream);

                static void launchProbeWriteKernel(Tuple *rightTuples, const uint64_t nRightTuple, uint64_t nLeftTuple,
                                                   Tuple *hashTable,
                                                   const unsigned long long int *histogram, const uint64_t *prefixSum,
                                                   unsigned long long int *resultCount, uint64_t numDistinctKeys,
                                                   cudaStream_t cudaStream, ResultTuple *sinkBuffer,
                                                   uint64_t sinkBufferSize);

                static void launchBuildHashTableKernel(Tuple *buildSideTuples, uint64_t nLeftTuples, Tuple *hashTable,
                                                       const uint64_t *prefixSum, unsigned long long *occupation,
                                                       uint64_t numDistinctKeys, uint64_t maxTupleInWindow,
                                                       cudaStream_t cudaStream);

                static void copyTuplesToLocalStore(Tuple *localStore, Tuple *ringBuffer, std::vector<uint64_t> indexes,
                                                   uint64_t batchSize, cudaStream_t cudaStream);

                static void copyTuplesToLocalStore(Tuple *localStore, Tuple *ringBuffer, std::vector<uint64_t> indexes,
                                                   uint64_t batchSize, const std::vector<cudaStream_t> &cudaStream);

                unsigned long long *h_histogram;
                unsigned long long *d_histogram;

                uint64_t *d_prefixSum;
                uint64_t *h_prefixSum;

                Tuple *d_hashTable;

                unsigned long long *d_occupation;

                // We need a dedicated non-page-locked GPU memory for result count to allow faster atomic operation
                unsigned long long *d_resultCount;
                unsigned long long *h_resultCount;

                std::vector<cudaStream_t> probeStreams;

                cudaStream_t leftStream;
                cudaStream_t rightStream;
                cudaStream_t joinerStream;

                uint64_t totalBuildTime = 0;
                uint64_t totalProbeTime = 0;
                uint64_t totalExecutedWindows = 0;

                Tuple *d_leftTuples;
                Tuple *d_rightTuples;

                uint64_t numDistinctKeys;
                uint64_t batchSize;

                uint64_t maxTupleInWindow;
            };
        }
    }
}

#endif //STREAMJOINBENCHMARK_GPU_HJ_CUH
