#include <sjb/windowing/lazy/executor/GPU/GPU_NLJ_CountKernel.cuh>
#include <sjb/utils/ErrorChecking.cuh>
#include <sjb/utils/Logger.hpp>
#include <sjb/sink/Sink.h>

namespace Windowing {
    namespace Lazy {
        namespace Executor {

            GPU_NLJ_CountKernel::~GPU_NLJ_CountKernel() = default;

            GPU_NLJ_CountKernel::GPU_NLJ_CountKernel(uint64_t batchSize, uint64_t maxTupleInWindow) : GPU_NLJ(batchSize,
                                                                                                        maxTupleInWindow) {

            }

            ExecutionStatistic GPU_NLJ_CountKernel::execute(Tuple *leftRingBuffer, Tuple *rightRingBuffer,
                                                         std::vector<uint64_t> leftIndexesToJoin,
                                                         std::vector<uint64_t> rightIndexesToJoin, Sink &sink) {
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

                // Launch the count kernel
                launchNestedLoopCountKernel(d_leftTuples, d_rightTuples, nLeftTuples,
                                            nRightTuples, d_resultCount, joinerStream);
                cudaStreamSynchronize(joinerStream);

                // Copy the result count to the host
                CUDA_CHECK(cudaMemcpyAsync(h_resultCount, d_resultCount, sizeof(unsigned long long),
                                           cudaMemcpyDeviceToHost,
                                           joinerStream));
                cudaStreamSynchronize(joinerStream);

                // Reset the result counter to 0
                CUDA_CHECK(cudaMemsetAsync(d_resultCount, 0, sizeof(unsigned long long), joinerStream));
                CUDA_CHECK(cudaStreamSynchronize(joinerStream));
                cudaStreamSynchronize(joinerStream);

                // Allocate output buffer of a size of maximum possible matches
                ResultTuple *resultTuple;
                LOG_DEBUG("Number of result tuple: %llu", *h_resultCount);
                CUDA_CHECK(cudaMallocHost(&resultTuple, *h_resultCount * sizeof(ResultTuple)));

                // Launch the join kernel
                launchNestedLoopJoinKernel(d_leftTuples, d_rightTuples, nLeftTuples,
                                            nRightTuples, d_resultCount, resultTuple, joinerStream);
                cudaStreamSynchronize(joinerStream);

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

                cudaFree(resultTuple);

                ExecutionStatistic es = ExecutionStatistic();
                return es;
            }
        }
    }
}