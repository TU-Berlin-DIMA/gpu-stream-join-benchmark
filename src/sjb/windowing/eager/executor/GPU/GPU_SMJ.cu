#include <windowing/eager/executor/GPU/GPU_SMJ.cuh>
#include <utils/CudaUtils.cuh>
#include <utils/ErrorChecking.cuh>
#include <utils/Logger.hpp>
#include <chrono>
#include <tbb/parallel_invoke.h>
#include "../../../../../../include/sjb/windowing/lazy/executor/GPU/GPU_SMJ.cuh"


#define PREFIX_SUM_THREAD_NUM 8
#define PREFIX_SUM_BLOCK_NUM 1
#define MAX_BLOCK_SZ 1024

// TODO: make a more elegant solution
//#define MAX_TUPLE_IN_WINDOW WINDOW_SIZE
#define EAGER_MAX_TUPLE_IN_WINDOW 10000

namespace Windowing {
    namespace Eager {
        namespace Executor {

            __global__ void smjSortKernel(Tuple *input, Tuple *output, uint32_t *prefixSum, uint64_t nTuples,
                                          uint32_t *occupation) {
                uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

                if (i < nTuples) {

                    auto &currentTuple = input[i];

                    auto oldCount = atomicAdd(&occupation[currentTuple.key], 1);
                    output[prefixSum[currentTuple.key] + oldCount].key = currentTuple.key;
                    output[prefixSum[currentTuple.key] + oldCount].val = currentTuple.val;
                    output[prefixSum[currentTuple.key] + oldCount].ts = currentTuple.ts;
                }
            }

            uint64_t GPU_SMJ::execute(Tuple *tupleBuffer, bool isLeftSide) {
                // copy the incoming tuple to gpu
                CUDA_CHECK(cudaMemcpyAsync(d_tuples, tupleBuffer, TUPLE_BUFFER_SIZE * sizeof(Tuple),
                                           cudaMemcpyHostToDevice, incomingTuplesMemcpyStream));
                computePrefixSum((d_tuples), TUPLE_BUFFER_SIZE, d_prefixSum);

                // sort the incoming tuple
                dim3 dimBlock(1024, 1, 1);
                dim3 dimGrid((TUPLE_BUFFER_SIZE + dimBlock.x - 1) / dimBlock.x, 1, 1);
                smjSortKernel<<<dimGrid, dimBlock, 0, sortStream>>>(d_tuples,
                                                                    d_tmp, d_prefixSum,
                                                                    TUPLE_BUFFER_SIZE, d_occupation);
                uint64_t count = 0;
                if (isLeftSide) {
                    tbb::parallel_invoke([&]() {
                        cudaStreamSynchronize(sortStream);

                        // append the sorted incoming tuple to its corresponding window
                        CUDA_CHECK(
                                cudaMemcpyAsync(d_leftWindow + leftWindowOccupation, d_tmp,
                                                TUPLE_BUFFER_SIZE * sizeof(Tuple),
                                                cudaMemcpyDeviceToDevice, sortedTuplesCopyStream));
                        CUDA_CHECK(cudaMemcpyAsync(d_leftWindowPrefixSums + leftWindowOccupation, d_prefixSum,
                                                   DISTINCT_KEYS * sizeof(uint32_t), cudaMemcpyDeviceToDevice,
                                                   prefixSumCopyStream));
                        leftWindowOccupation += TUPLE_BUFFER_SIZE;

                    }, [&]() {
                        // merge with the other side
                        merge(d_tuples, d_prefixSum, d_rightWindow, d_rightWindowPrefixSums,
                              rightWindowOccupation);
                    });


                } else {
                    tbb::parallel_invoke([&]() {
                        cudaStreamSynchronize(sortStream);

                        // append the sorted incoming tuple to its corresponding window
                        CUDA_CHECK(cudaMemcpyAsync(d_rightWindow + rightWindowOccupation, d_tuples,
                                                   TUPLE_BUFFER_SIZE * sizeof(Tuple), cudaMemcpyDeviceToDevice,
                                                   sortedTuplesCopyStream));
                        CUDA_CHECK(cudaMemcpyAsync(d_rightWindowPrefixSums + rightWindowOccupation, d_prefixSum,
                                                   DISTINCT_KEYS * sizeof(uint32_t), cudaMemcpyDeviceToDevice,
                                                   prefixSumCopyStream));
                        rightWindowOccupation += TUPLE_BUFFER_SIZE;

                    }, [&]() {
                        // merge with the other side
                        merge(d_tuples, d_prefixSum, d_leftWindow, d_leftWindowPrefixSums,
                              leftWindowOccupation);
                    });
                }

                // copy the match count
                cudaStreamSynchronize(mergeStream);
                CUDA_CHECK(
                        cudaMemcpyAsync(h_matchCount, d_matchCount, sizeof(unsigned long long int),
                                        cudaMemcpyDeviceToHost, matchCountMemcpyStream));

                // synchronize streams
                cudaStreamSynchronize(sortedTuplesCopyStream);
                cudaStreamSynchronize(prefixSumCopyStream);
                cudaStreamSynchronize(matchCountMemcpyStream);
                LOG_DEBUG("count before: %lu", count);
                count += *h_matchCount;
                LOG_DEBUG("count after: %lu", count);
                return count;
            }

            void GPU_SMJ::clearStates() {
                LOG_INFO("clearStates()");
                leftWindowOccupation = 0;
                rightWindowOccupation = 0;

                CUDA_CHECK(cudaMemsetAsync(d_leftWindowPrefixSums, 0,
                                           (EAGER_MAX_TUPLE_IN_WINDOW / TUPLE_BUFFER_SIZE) * DISTINCT_KEYS * sizeof(uint32_t),
                                           lWindowPrefixSumMemsetStream));
                CUDA_CHECK(cudaMemsetAsync(d_rightWindowPrefixSums, 0,
                                           (EAGER_MAX_TUPLE_IN_WINDOW / TUPLE_BUFFER_SIZE) * DISTINCT_KEYS * sizeof(uint32_t),
                                           rWindowPrefixSumMemsetStream));
                CUDA_CHECK(cudaMemsetAsync(d_histogram, 0, DISTINCT_KEYS * sizeof(uint32_t), histogramMemsetStream));
                CUDA_CHECK(cudaMemsetAsync(d_matchCount, 0, sizeof(unsigned long long int), matchCountMemsetStream));

                cudaStreamSynchronize(lWindowPrefixSumMemsetStream);
                cudaStreamSynchronize(rWindowPrefixSumMemsetStream);
                cudaStreamSynchronize(histogramMemsetStream);
                cudaStreamSynchronize(matchCountMemsetStream);
            }

            GPU_SMJ::~GPU_SMJ() {
                CUDA_CHECK(cudaFree(d_leftWindow));
                CUDA_CHECK(cudaFree(d_rightWindow));

                CUDA_CHECK(cudaFree(d_leftWindowPrefixSums));
                CUDA_CHECK(cudaFree(d_rightWindowPrefixSums));

                CUDA_CHECK(cudaFree(d_tmp));
                CUDA_CHECK(cudaFree(d_occupation));

                CUDA_CHECK(cudaFree(d_tuples));
                CUDA_CHECK(cudaFree(d_prefixSum));

                CUDA_CHECK(cudaFree(d_histogram));
                CUDA_CHECK(cudaFree(d_matchCount));

                cudaStreamDestroy(mergeStream);
                cudaStreamDestroy(sortStream);
                cudaStreamDestroy(sortedTuplesCopyStream);
                cudaStreamDestroy(prefixSumCopyStream);
                cudaStreamDestroy(prefixSumStream);
                cudaStreamDestroy(histogramStream);
                cudaStreamDestroy(matchCountMemcpyStream);
                cudaStreamDestroy(incomingTuplesMemcpyStream);

                cudaStreamDestroy(lWindowPrefixSumMemsetStream);
                cudaStreamDestroy(rWindowPrefixSumMemsetStream);
                cudaStreamDestroy(histogramMemsetStream);
                cudaStreamDestroy(matchCountMemsetStream);

            }

            GPU_SMJ::GPU_SMJ() {
                uint64_t nBufferInWindow = EAGER_MAX_TUPLE_IN_WINDOW/TUPLE_BUFFER_SIZE; // in time-based, we can't guarantee the num of tuples in a window
                CUDA_CHECK(cudaMalloc(&d_leftWindow, EAGER_MAX_TUPLE_IN_WINDOW * sizeof(Tuple)));
                CUDA_CHECK(cudaMalloc(&d_rightWindow, EAGER_MAX_TUPLE_IN_WINDOW * sizeof(Tuple)));

                CUDA_CHECK(cudaMalloc(&d_leftWindowPrefixSums,
                                      nBufferInWindow * DISTINCT_KEYS * sizeof(uint32_t)));
                CUDA_CHECK(cudaMalloc(&d_rightWindowPrefixSums,
                                      nBufferInWindow * DISTINCT_KEYS * sizeof(uint32_t)));

                CUDA_CHECK(cudaMalloc(&d_tmp, TUPLE_BUFFER_SIZE * sizeof(Tuple)));
                CUDA_CHECK(cudaMalloc(&d_occupation, DISTINCT_KEYS * sizeof(uint32_t)));
                CUDA_CHECK(cudaMemset(d_occupation, 0, DISTINCT_KEYS * sizeof(uint32_t)));

                CUDA_CHECK(cudaMallocHost(&h_matchCount, sizeof(unsigned long long int)));

                CUDA_CHECK(cudaMalloc(&d_tuples, TUPLE_BUFFER_SIZE * sizeof(Tuple)));
                CUDA_CHECK(cudaMemset(d_tuples, 0, TUPLE_BUFFER_SIZE * sizeof(Tuple)));

                CUDA_CHECK(cudaMalloc(&d_prefixSum, DISTINCT_KEYS * sizeof(uint32_t)));

                CUDA_CHECK(cudaMalloc(&d_histogram, DISTINCT_KEYS * sizeof(uint32_t)));
                CUDA_CHECK(cudaMemset(d_histogram, 0, DISTINCT_KEYS * sizeof(uint32_t)));

                CUDA_CHECK(cudaMalloc(&d_matchCount, sizeof(unsigned long long int)));
                CUDA_CHECK(cudaMemset(d_matchCount, 0, sizeof(unsigned long long int)));

                cudaStreamCreate(&mergeStream);
                cudaStreamCreate(&sortStream);
                cudaStreamCreate(&sortedTuplesCopyStream);
                cudaStreamCreate(&prefixSumCopyStream);
                cudaStreamCreate(&prefixSumStream);
                cudaStreamCreate(&histogramStream);
                cudaStreamCreate(&matchCountMemcpyStream);
                cudaStreamCreate(&incomingTuplesMemcpyStream);

                cudaStreamCreate(&lWindowPrefixSumMemsetStream);
                cudaStreamCreate(&rWindowPrefixSumMemsetStream);
                cudaStreamCreate(&histogramMemsetStream);
                cudaStreamCreate(&matchCountMemsetStream);
            }


            __global__ void histogramKernel(Tuple *d_tuples, uint64_t size, uint32_t *d_histogram) {
                uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

                if (i < size) {
                    atomicAdd(&d_histogram[d_tuples[i].key], 1);
                }
            }

            __global__ void prescan(uint32_t *g_odata, const uint32_t *g_idata, int n) {
                extern  __shared__  uint32_t temp[];
                // allocated on invocation
                uint32_t thid = threadIdx.x;
                uint32_t bid = blockIdx.x;

                int offset = 1;
                if ((bid * PREFIX_SUM_THREAD_NUM + thid) < n) {
                    temp[thid] = g_idata[bid * PREFIX_SUM_THREAD_NUM + thid];
                } else {
                    temp[thid] = 0;
                } // Make the "empty" spots zeros, so it won't affect the final result.

                for (uint32_t d = PREFIX_SUM_THREAD_NUM >> 1; d > 0; d >>= 1) {
                    // build sum in place up the tree
                    __syncthreads();
                    if (thid < d) {
                        uint32_t ai = offset * (2 * thid + 1) - 1;
                        uint32_t bi = offset * (2 * thid + 2) - 1;
                        temp[bi] += temp[ai];
                    }
                    offset *= 2;
                }

                if (thid == 0) {
                    temp[PREFIX_SUM_THREAD_NUM - 1] = 0;
                }

                // clear the last element
                for (int d = 1; d < PREFIX_SUM_THREAD_NUM; d *= 2)
                    // traverse down tree & build scan
                {
                    offset >>= 1;
                    __syncthreads();
                    if (thid < d) {
                        uint32_t ai = offset * (2 * thid + 1) - 1;
                        uint32_t bi = offset * (2 * thid + 2) - 1;
                        uint32_t t = temp[ai];
                        temp[ai] = temp[bi];
                        temp[bi] += t;
                    }
                }
                __syncthreads();
                g_odata[bid * PREFIX_SUM_THREAD_NUM + thid] = temp[thid];
            }

            void GPU_SMJ::computePrefixSum(Tuple *d_tuples, uint64_t size, uint32_t *d_prefixSum) {
                dim3 dimBlock(1024, 1, 1);
                dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x, 1, 1);

                histogramKernel<<<dimGrid, dimBlock, 0, histogramStream>>>(d_tuples, size, d_histogram);
                cudaStreamSynchronize(histogramStream);
                prescan<<<PREFIX_SUM_BLOCK_NUM, PREFIX_SUM_THREAD_NUM, 2 * PREFIX_SUM_THREAD_NUM *
                                                                       sizeof(float), prefixSumStream>>>(
                        d_prefixSum, d_histogram,
                        DISTINCT_KEYS);
                cudaDeviceSynchronize();
            }

            __global__ void smjMergeKernel(Tuple *d_incomingTuples, Tuple *d_window, uint32_t *d_incomingTuplePrefixSum,
                                           uint32_t *d_windowPrefixSums, unsigned long long int windowOccupation,
                                           unsigned long long int *d_matchCount) {
                uint64_t runIdx = blockIdx.x * blockDim.x + threadIdx.x;
                uint64_t key = blockIdx.y * blockDim.y + threadIdx.y;


                if (runIdx < windowOccupation / TUPLE_BUFFER_SIZE && key < DISTINCT_KEYS - 1) {
                    Tuple *incomingTupleKeyBucket = &d_incomingTuples[d_incomingTuplePrefixSum[key]];

                    Tuple *currentRun = &d_window[runIdx * TUPLE_BUFFER_SIZE];
                    uint32_t *currentRunPrefixSum = &d_windowPrefixSums[runIdx + TUPLE_BUFFER_SIZE];
                    Tuple *currentRunKeyBucket = &currentRun[currentRunPrefixSum[key]];

                    unsigned long long int localCount = 0;
                    for (uint64_t i = d_incomingTuplePrefixSum[key]; i < d_incomingTuplePrefixSum[key + 1]; i++) {
                        for (uint64_t j = currentRunPrefixSum[key]; j < currentRunPrefixSum[key + 1]; j++) {
                            // result.key = key
                            // result.lVal = incomingTupleKeyBucket[i].val
                            // result.rVal = currentRunKeyBucket[j].val
                            // result.lTs = incomingTupleKeyBucket[i].ts
                            // result.rTs = currentRunKeyBucket[j].ts
                            localCount++;
                        }
                    }
                    for (uint64_t c = 0; c < localCount; c++) {
                        CudaUtils::atomicAggInc(d_matchCount);
                    }
                }
            }

            void GPU_SMJ::merge(Tuple *d_incomingTuples, uint32_t *d_incomingTuplePrefixSum, Tuple *d_window,
                                uint32_t *d_windowPrefixSums, unsigned long long int windowOccupation) {

                dim3 dimBlock(32, 32);
                dim3 dimGrid;
                if (windowOccupation < TUPLE_BUFFER_SIZE) {
                    dimGrid.x = 1;
                } else {
                    dimGrid.x = (windowOccupation / TUPLE_BUFFER_SIZE + dimBlock.x - 1) / dimBlock.x;

                }
                dimGrid.y = (DISTINCT_KEYS + dimBlock.y - 1) / dimBlock.y;

                smjMergeKernel<<<dimGrid, dimBlock, 0, mergeStream>>>(d_incomingTuples, d_window,
                                                                      d_incomingTuplePrefixSum,
                                                                      d_windowPrefixSums, windowOccupation,
                                                                      d_matchCount);
            }
        }
    }
}