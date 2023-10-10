#include <atomic>
#include <chrono>

#include <sjb/windowing/lazy/executor/GPU/GPU_SMJ_EstimatedSelectivity.cuh>
#include <sjb/utils/ErrorChecking.cuh>
#include <sjb/utils/Logger.hpp>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <sjb/utils/ResultTuple.hpp>
#include <sjb/sink/Sink.h>
#include <sjb/utils/CudaUtils.cuh>

#define xstr(a) str(a)
#define str(a) #a

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            GPU_SMJ_EstimatedSelectivity::GPU_SMJ_EstimatedSelectivity(uint64_t numDistinctKeys, uint64_t batchSize,
                                                                       uint64_t maxTupleInWindow,
                                                                       double estimatedSelectivity) :
                    GPU_SMJ(numDistinctKeys, batchSize, maxTupleInWindow),
                    estimatedSelectivity(estimatedSelectivity) {
            }

            GPU_SMJ_EstimatedSelectivity::~GPU_SMJ_EstimatedSelectivity() {
            }


            ExecutionStatistic
            GPU_SMJ_EstimatedSelectivity::execute(Tuple *leftRingBuffer, Tuple *rightRingBuffer,
                                                  std::vector<uint64_t> leftIndexesToCopy,
                                                  std::vector<uint64_t> rightIndexesToCopy, Sink &sink) {
                auto t0 = std::chrono::high_resolution_clock::now();

                auto nLeftTuples = leftIndexesToCopy.size() * batchSize;
                auto nRightTuples = rightIndexesToCopy.size() * batchSize;

                CUDA_CHECK(
                        cudaMemsetAsync(d_leftOccupation, 0, numDistinctKeys * sizeof(unsigned long long), leftStream));
                CUDA_CHECK(cudaMemsetAsync(d_rightOccupation, 0, numDistinctKeys * sizeof(unsigned long long),
                                           rightStream));

                CUDA_CHECK(cudaMemsetAsync(d_leftHistogram, 0, numDistinctKeys * sizeof(uint64_t), leftStream));
                CUDA_CHECK(cudaMemsetAsync(d_rightHistogram, 0, numDistinctKeys * sizeof(uint64_t), rightStream));

                // Copy data from ring buffer to the GPU
                copyTuplesToLocalStore(d_leftTuples, leftRingBuffer, leftIndexesToCopy, batchSize, leftStream);
                CUDA_CHECK(cudaStreamSynchronize(leftStream));

                launchHistogramKernel(d_leftTuples, nLeftTuples, d_leftHistogram,leftStream);
                CUDA_CHECK(cudaStreamSynchronize(leftStream));


                auto leftPolicy = thrust::cuda::par.on(leftStream);
                thrust::exclusive_scan(leftPolicy, d_leftHistogram, d_leftHistogram + numDistinctKeys, d_leftPrefixSum);

                launchSortKernel(d_leftTuples,
                                 d_sortedLeftTuples,
                                 d_leftPrefixSum,
                                 nLeftTuples,
                                 d_leftOccupation,
                                 leftStream);
                CUDA_CHECK(cudaStreamSynchronize(leftStream));

                copyTuplesToLocalStore(d_rightTuples, rightRingBuffer, rightIndexesToCopy, batchSize, rightStream);
                CUDA_CHECK(cudaStreamSynchronize(rightStream));

                launchHistogramKernel(d_rightTuples, nRightTuples, d_rightHistogram, rightStream);
                CUDA_CHECK(cudaStreamSynchronize(rightStream));

                auto rightPolicy = thrust::cuda::par.on(rightStream);
                thrust::exclusive_scan(rightPolicy, d_rightHistogram, d_rightHistogram + numDistinctKeys,
                                       d_rightPrefixSum);

                launchSortKernel(d_rightTuples, d_sortedRightTuples, d_rightPrefixSum, nRightTuples, d_rightOccupation, rightStream);
                CUDA_CHECK(cudaStreamSynchronize(rightStream));


                CUDA_CHECK(cudaStreamSynchronize(leftStream));
                CUDA_CHECK(cudaStreamSynchronize(rightStream));

                auto t1 = std::chrono::high_resolution_clock::now();



                // Allocate output buffer of a size of maximum possible matches
                // Allocate twice the estimated result to allow for minor error in the estimation
                ResultTuple *d_resultTuple;
                uint64_t expectedResultCount = 1.125 * ceil(estimatedSelectivity * nLeftTuples * nRightTuples);
                LOG_DEBUG("Expected result count: %lu", expectedResultCount);
                CUDA_CHECK(cudaMalloc(&d_resultTuple, expectedResultCount * sizeof(ResultTuple)));

                // Reset the result counter to 0 before joining
                CUDA_CHECK(cudaMemsetAsync(d_resultCount, 0, sizeof(unsigned long long), joinerStream));
                CUDA_CHECK(cudaStreamSynchronize(joinerStream));

                launchMergeJoinKernel(d_leftHistogram, d_rightHistogram, d_sortedLeftTuples,
                                      d_sortedRightTuples, d_resultCount, numDistinctKeys, d_resultTuple, joinerStream);
                CUDA_CHECK(cudaStreamSynchronize(joinerStream));

                CUDA_CHECK(cudaMemcpyAsync(h_resultCount, d_resultCount, sizeof(unsigned long long),
                                           cudaMemcpyDeviceToHost,
                                           joinerStream));
                CUDA_CHECK(cudaStreamSynchronize(joinerStream));

                uint64_t nWrittenTuple = 0;
                uint64_t copyCount;
                while (nWrittenTuple < *h_resultCount) {
                    if (nWrittenTuple + sink.getSinkBufferSize() > *h_resultCount) {
                        copyCount = *h_resultCount-nWrittenTuple;
                    } else {
                        copyCount = sink.getSinkBufferSize();
                    }
                    sink.incrementCounterAndStore(d_resultTuple+nWrittenTuple, copyCount);
                    nWrittenTuple += copyCount;
                }
                assert(nWrittenTuple == *h_resultCount);

                auto t2 = std::chrono::high_resolution_clock::now();

                totalSortTime += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                totalMergeTime += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
                totalExecutedWindows++;

                ExecutionStatistic es = ExecutionStatistic();
                es.sortTime = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();;
                es.mergeTime = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();;

                cudaFree(d_resultTuple);
                return es;
            }
        }
    }
}
