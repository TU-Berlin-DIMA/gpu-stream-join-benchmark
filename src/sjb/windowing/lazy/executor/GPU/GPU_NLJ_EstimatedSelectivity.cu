#include <sjb/windowing/lazy/executor/GPU/GPU_NLJ_EstimatedSelectivity.cuh>
#include <sjb/utils/ErrorChecking.cuh>
#include <sjb/utils/Logger.hpp>
#include <sjb/sink/Sink.h>

namespace Windowing {
    namespace Lazy {
        namespace Executor {

            GPU_NLJ_EstimatedSelectivity::~GPU_NLJ_EstimatedSelectivity() = default;

            GPU_NLJ_EstimatedSelectivity::GPU_NLJ_EstimatedSelectivity(uint64_t batchSize, uint64_t maxTupleInWindow,double estimatedSelectivity) : GPU_NLJ(batchSize,
                                                                                                              maxTupleInWindow),
                                                                                                      estimatedSelectivity(
                                                                                                              estimatedSelectivity) {

            }

            ExecutionStatistic GPU_NLJ_EstimatedSelectivity::execute(Tuple *leftRingBuffer, Tuple *rightRingBuffer,
                                                                     std::vector<uint64_t> leftIndexesToJoin,
                                                                     std::vector<uint64_t> rightIndexesToJoin,
                                                                     Sink &sink) {
                uint64_t nLeftTuples = leftIndexesToJoin.size() * batchSize;
                uint64_t nRightTuples = rightIndexesToJoin.size() * batchSize;

                // Copy tuples to the GPU
                copyTuplesToLocalStore(d_leftTuples, leftRingBuffer, leftIndexesToJoin, batchSize, leftStream);
                copyTuplesToLocalStore(d_rightTuples, rightRingBuffer, rightIndexesToJoin, batchSize, rightStream);

                cudaStreamSynchronize(leftStream);
                cudaStreamSynchronize(rightStream);

                // Reset the result counter to 0
                CUDA_CHECK(cudaMemsetAsync(d_resultCount, 0, sizeof(unsigned long long), joinerStream));
                CUDA_CHECK(cudaStreamSynchronize(joinerStream));

                // Allocate output buffer of a size of maximum possible matches
                // Allocate ten times the estimated result to allow for errors in the estimation
                ResultTuple *resultTuple;
                uint64_t expectedResultCount = 10 * ceil(estimatedSelectivity * nLeftTuples * nRightTuples);
                LOG_DEBUG("ES: %f, nLeft:%lu, nRight:%lu, Expected result count: %lu",estimatedSelectivity, nLeftTuples, nRightTuples, expectedResultCount);
                CUDA_CHECK(cudaMallocHost(&resultTuple, expectedResultCount * sizeof(ResultTuple)));

                // Reset the result counter to 0
                CUDA_CHECK(cudaMemsetAsync(d_resultCount, 0, sizeof(unsigned long long), joinerStream));
                CUDA_CHECK(cudaStreamSynchronize(joinerStream));
                cudaStreamSynchronize(joinerStream);

                // Launch the join kernel
                launchNestedLoopJoinKernel(d_leftTuples, d_rightTuples, nLeftTuples,
                                           nRightTuples, d_resultCount, resultTuple, joinerStream);
                cudaStreamSynchronize(joinerStream);

                // Copy the result count back to host
                CUDA_CHECK(cudaMemcpyAsync(h_resultCount, d_resultCount, sizeof(unsigned long long),
                                           cudaMemcpyDeviceToHost,
                                           joinerStream));
                CUDA_CHECK(cudaStreamSynchronize(joinerStream));
                LOG_DEBUG("Actual resultCount:%llu", *h_resultCount);

                // Copy the join result to the sink
                for (uint64_t resultIndex = 0; resultIndex < *h_resultCount; resultIndex++) {
                    // TODO: copy in bulk instead of on a tuple basis
                    sink.incrementCounterAndStore(resultTuple->lkey,
                                                  resultTuple->rkey,
                                                  resultTuple->lVal,
                                                  resultTuple->rVal,
                                                  resultTuple->lTs,
                                                  resultTuple->rTs
                    );
                }

                // Clean up the copied result tuple
                cudaFreeHost(resultTuple);

                ExecutionStatistic es = ExecutionStatistic();
                return es;
            }
        }
    }
}