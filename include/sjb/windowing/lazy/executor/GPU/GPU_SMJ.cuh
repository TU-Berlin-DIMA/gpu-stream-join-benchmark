#ifndef STREAMJOINBENCHMARK_GPU_SMJ_CUH
#define STREAMJOINBENCHMARK_GPU_SMJ_CUH

#include <sjb/windowing/lazy/executor/BaseLazyExecutor.hpp>
#include <sjb/utils/ResultTuple.hpp>

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            class GPU_SMJ : public BaseLazyExecutor {
            public:
                GPU_SMJ(uint64_t numDistinctKeys, uint64_t batchSize,
                        uint64_t maxTupleInWindow);

                ExecutionStatistic execute(Tuple *leftRingBuffer, Tuple *rightRingBuffer,
                                           std::vector<uint64_t> leftIndexesToCopy,
                                           std::vector<uint64_t> rightIndexesToCopy,
                                           Sink &sink) override = 0;

                static void launchHistogramKernel(Tuple *buildSide,
                                           uint64_t tupleCounts,
                                           unsigned long long int *histogram,
                                           cudaStream_t cudaStream);

                static void launchSortKernel(Tuple *input, Tuple *output, uint64_t *prefixSum, uint64_t nTuples,
                                      unsigned long long *occupation, cudaStream_t cudaStream);

                static void launchMergeCountKernel(const unsigned long long int *leftHistogram,
                                            const unsigned long long int *rightHistogram,
                                            unsigned long long *resultCount,
                                            uint64_t numDistinctKeys, cudaStream_t cudaStream);

                static void launchMergeWriteKernel(const unsigned long long int *leftHistogram,
                                                   const unsigned long long int *rightHistogram,
                                                   Tuple *sortedLeftTuples,
                                                   Tuple *sortedRightTuples,
                                                   unsigned long long int *resultCount,
                                                   uint64_t numDistinctKeys,
                                                   cudaStream_t cudaStream,
                                                   ResultTuple *sinkBuffer,
                                                   uint64_t sinkBufferSize);

                static void launchMergeJoinKernel(const unsigned long long int *leftHistogram,
                                                  const unsigned long long int *rightHistogram,
                                                  Tuple *sortedLeftTuples,
                                                  Tuple *sortedRightTuples,
                                                  unsigned long long *resultCount,
                                                  uint64_t numDistinctKeys,
                                                  ResultTuple *resultTuple,
                                                  cudaStream_t cudaStream);

                static void copyTuplesToLocalStore(Tuple *localStore, Tuple *ringBuffer, std::vector<uint64_t> indexes,
                uint64_t batchSize,cudaStream_t cudaStream);

                ~GPU_SMJ();

            protected:
                // store left and right tuples
                Tuple *d_leftTuples{};
                Tuple *d_rightTuples{};

                // store left and right tuples
                Tuple *d_sortedLeftTuples{};
                Tuple *d_sortedRightTuples{};

                // stores the d_histogram from both side
                unsigned long long int *d_leftHistogram;
                unsigned long long int *d_rightHistogram;

                unsigned long long *d_leftOccupation;
                unsigned long long *d_rightOccupation;

                uint64_t *d_leftPrefixSum;
                uint64_t *d_rightPrefixSum;

                unsigned long long int *d_resultCount;
                unsigned long long int *h_resultCount;

                cudaStream_t leftStream;
                cudaStream_t rightStream;
                cudaStream_t joinerStream;

                uint64_t totalSortTime = 0;
                uint64_t totalMergeTime = 0;
                uint64_t totalExecutedWindows = 0;

                uint64_t numDistinctKeys;
                uint64_t batchSize;

                uint64_t maxTupleInWindow;
            };
        }
    }
}
#endif //STREAMJOINBENCHMARK_GPU_SMJ_CUH
