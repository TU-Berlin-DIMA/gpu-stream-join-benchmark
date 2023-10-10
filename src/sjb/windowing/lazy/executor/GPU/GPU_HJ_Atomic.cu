#include <chrono>
#include <algorithm>
#include <sjb/windowing/lazy/executor/GPU/GPU_HJ_Atomic.cuh>
#include <sjb/sink/Sink.h>
#include <sjb/utils/ErrorChecking.cuh>
#include <sjb/utils/Logger.hpp>
#include <cassert>

#define xstr(a) str(a)
#define str(a) #a

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            GPU_HJ_Atomic::GPU_HJ_Atomic(
                    uint64_t numDistinctKeys,
                    uint64_t batchSize,
                    uint64_t maxTupleInWindow) : GPU_HJ(numDistinctKeys, batchSize, maxTupleInWindow) {

            }

            ExecutionStatistic GPU_HJ_Atomic::execute(Tuple *leftRingBuffer, Tuple *rightRingBuffer,
                                                        std::vector<uint64_t> leftIndexesToJoin,
                                                        std::vector<uint64_t> rightIndexesToJoin,
                                                        Sink &sink) {
                auto nLeftTuples = leftIndexesToJoin.size() * batchSize;
                auto nRightTuples = rightIndexesToJoin.size() * batchSize;

                // Assert that we have enough hash table
                LOG_DEBUG("maxTupleInWindow: %lu, leftIndexesToJoin.size(): %lu, batchSize:%lu", maxTupleInWindow,
                          leftIndexesToJoin.size(), batchSize);
                assert(maxTupleInWindow >= nLeftTuples);

                // Start the timer
                auto t0 = std::chrono::high_resolution_clock::now();

                // Copy the build side to the local store of the joiner
                copyTuplesToLocalStore(d_leftTuples, leftRingBuffer, leftIndexesToJoin, batchSize, leftStream);
                copyTuplesToLocalStore(d_rightTuples, rightRingBuffer, rightIndexesToJoin, batchSize, rightStream);

                CUDA_CHECK(cudaStreamSynchronize(leftStream));

                // Reset the histogram
                CUDA_CHECK(cudaMemsetAsync(d_histogram, 0, numDistinctKeys * sizeof(unsigned long long), joinerStream));
                CUDA_CHECK(cudaStreamSynchronize(joinerStream));

                launchComputeHistogramKernel(d_leftTuples, nLeftTuples, d_histogram, numDistinctKeys, joinerStream);
                CUDA_CHECK(cudaStreamSynchronize(joinerStream));

                // Copy the computed histogram back to host
                CUDA_CHECK(cudaMemcpyAsync(h_histogram, d_histogram, numDistinctKeys * sizeof(unsigned  long long), cudaMemcpyDeviceToHost, joinerStream));
                CUDA_CHECK(cudaStreamSynchronize(joinerStream));

                // Compute prefix sum from the histogram
                memset(h_prefixSum, 0, numDistinctKeys * sizeof(uint64_t));
                uint64_t runningSum = 0;
                for (uint64_t kIdx = 0; kIdx < numDistinctKeys; kIdx++) {
                    h_prefixSum[kIdx] = runningSum;
                    runningSum += h_histogram[kIdx];
                }

                // Copy the prefixSum to the GPU
                CUDA_CHECK(cudaMemcpyAsync(d_prefixSum, h_prefixSum, numDistinctKeys * sizeof(uint64_t), cudaMemcpyHostToDevice, joinerStream));
                CUDA_CHECK(cudaStreamSynchronize(joinerStream));

                // Reset the occupation manager
                CUDA_CHECK(cudaMemsetAsync(d_occupation, 0, numDistinctKeys * sizeof(unsigned long long), joinerStream));
                CUDA_CHECK(cudaStreamSynchronize(joinerStream));

                // Build the hashtable
                launchBuildHashTableKernel(d_leftTuples, nLeftTuples, d_hashTable,
                                           d_prefixSum, d_occupation, numDistinctKeys, maxTupleInWindow,
                                           joinerStream);
                CUDA_CHECK(cudaStreamSynchronize(joinerStream));

                // Reset the result counter to 0
                CUDA_CHECK(cudaMemsetAsync(d_resultCount, 0, sizeof(unsigned long long), joinerStream));
                CUDA_CHECK(cudaStreamSynchronize(joinerStream));

                // End of build time
                auto t1 = std::chrono::high_resolution_clock::now();

                // Make sure that the probe-side tuple has arrived to the GPU
                CUDA_CHECK(cudaStreamSynchronize(rightStream));


                // Launch the probe count kernel
                launchProbeWriteKernel(d_rightTuples, nRightTuples, nLeftTuples,
                                       d_hashTable, d_histogram, d_prefixSum,
                                       sink.counterGPU, numDistinctKeys, joinerStream,
                                       sink.sinkBuffer, sink.sinkBufferSize);
                CUDA_CHECK(cudaStreamSynchronize(joinerStream));

                // End of probe time
                auto t2 = std::chrono::high_resolution_clock::now();

                // Write the result count to experiment statistics
                ExecutionStatistic es = ExecutionStatistic();
                es.resultCount = *sink.counterGPU;

                LOG_DEBUG("Build time: %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
                LOG_DEBUG("Probe time: %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
                LOG_DEBUG("Counter: %llu", *h_resultCount);

                auto buildTime = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                auto probeTime = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

                totalBuildTime += buildTime;
                totalProbeTime += probeTime;
                totalExecutedWindows++;

                es.buildTime = buildTime;
                es.probeTime = probeTime;
                return es;
            }

            GPU_HJ_Atomic::~GPU_HJ_Atomic() = default;
        }
    }
}