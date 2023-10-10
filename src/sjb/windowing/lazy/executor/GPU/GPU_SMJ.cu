#include <sjb/windowing/lazy/executor/GPU/GPU_SMJ.cuh>
#include <sjb/utils/ErrorChecking.cuh>
#include <sjb/utils/Logger.hpp>
#include <sjb/utils/ResultTuple.hpp>

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            __global__ void smjHistogramKernel(Tuple *buildSide,
                                               const uint64_t tupleCounts,
                                               unsigned long long int *histogram) {
                uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

                if (i < tupleCounts) {
                    atomicAdd(&histogram[buildSide[i].key], 1);
                }
            }

            __global__ void smjSortKernel(Tuple *input, Tuple *output, uint64_t *prefixSum, uint64_t nTuples,
                                          unsigned long long *occupation) {
                uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

                if (i < nTuples) {

                    auto &currentTuple = input[i];

                    auto oldCount = atomicAdd(&occupation[currentTuple.key], 1);
                    output[prefixSum[currentTuple.key] + oldCount].key = currentTuple.key;
                    output[prefixSum[currentTuple.key] + oldCount].val = currentTuple.val;
                    output[prefixSum[currentTuple.key] + oldCount].ts = currentTuple.ts;
                }
            }

            __global__ void smjMergeKernelCount(const unsigned long long int *leftHistogram,
                                                const unsigned long long int *rightHistogram,
                                                unsigned long long *resultCount,
                                                uint64_t numDistinctKeys) {
                uint64_t keyIdx = blockIdx.x * blockDim.x + threadIdx.x;
                if (keyIdx < numDistinctKeys) {
                    if (leftHistogram[keyIdx] > 0 && rightHistogram[keyIdx] > 0) {
                        atomicAdd(resultCount, (leftHistogram[keyIdx] * rightHistogram[keyIdx]));
                    }
                }
            }

            __global__ void smjMergeKernelWrite(const unsigned long long int *leftHistogram,
                                                const unsigned long long int *rightHistogram,
                                                Tuple *sortedLeftTuples,
                                                Tuple *sortedRightTuples,
                                                unsigned long long *resultCount,
                                                uint64_t numDistinctKeys,
                                                ResultTuple *sinkBuffer,
                                                uint64_t sinkBufferSize) {
                uint64_t keyIdx = blockIdx.x * blockDim.x + threadIdx.x;
                if (keyIdx < numDistinctKeys) {
                    if (leftHistogram[keyIdx] > 0 && rightHistogram[keyIdx] > 0) {
                        auto writePos = atomicAdd(resultCount, (leftHistogram[keyIdx] * rightHistogram[keyIdx]));

                        // write
                        for (uint64_t lidx = 0; lidx < leftHistogram[keyIdx]; lidx++) {
                            for (uint64_t ridx = 0; ridx < rightHistogram[keyIdx]; ridx++) {
                                sinkBuffer[writePos % sinkBufferSize].lkey = keyIdx;
                                sinkBuffer[writePos % sinkBufferSize].rkey = keyIdx;
                                sinkBuffer[writePos % sinkBufferSize].lVal = sortedLeftTuples[lidx].val;
                                sinkBuffer[writePos % sinkBufferSize].lTs = sortedLeftTuples[lidx].ts;
                                sinkBuffer[writePos % sinkBufferSize].rVal = sortedRightTuples[ridx].val;
                                sinkBuffer[writePos % sinkBufferSize].rTs = sortedRightTuples[ridx].ts;
                                writePos++;
                            }
                        }
                    }
                }
            }


            /**
            * @brief find the join matches from two sorted array
            * @assumption the number of distinct key is known, the distinct key range from [0, DK]
            * @param leftHistogram
            * @param rightHistogram
            * @param resultCount
            * @param numDistinctKeys
            * @param resultTuple
            */
            __global__ void smjMergeKernel_ES(const unsigned long long int *leftHistogram,
                                              const unsigned long long int *rightHistogram,
                                              Tuple *sortedLeftTuples,
                                              Tuple *sortedRightTuples,
                                              unsigned long long *resultCount,
                                              uint64_t numDistinctKeys,
                                              ResultTuple *resultTuple) {
                uint64_t keyIdx = blockIdx.x * blockDim.x + threadIdx.x;
                if (keyIdx < numDistinctKeys) {
//                    printf("leftHistogram:%llu rightHistogram:%llu, LxR=%llu \n", leftHistogram[keyIdx],
//                           rightHistogram[keyIdx], leftHistogram[keyIdx] * rightHistogram[keyIdx]);

                    if (leftHistogram[keyIdx] > 0 && rightHistogram[keyIdx] > 0) {
                        auto nReservedResult = leftHistogram[keyIdx] * rightHistogram[keyIdx];
                        auto writePos = atomicAdd(resultCount, nReservedResult);
                        for (uint64_t lidx = 0; lidx < leftHistogram[keyIdx]; lidx++) {
                            for (uint64_t ridx = 0; ridx < rightHistogram[keyIdx]; ridx++) {
                                resultTuple[writePos].lkey = keyIdx;
                                resultTuple[writePos].rkey = keyIdx;
                                resultTuple[writePos].lVal = sortedLeftTuples[lidx].val;
                                resultTuple[writePos].lTs = sortedLeftTuples[lidx].ts;
                                resultTuple[writePos].rVal = sortedRightTuples[ridx].val;
                                resultTuple[writePos].rTs = sortedRightTuples[ridx].ts;
                                writePos++;
                            }
                        }
                    }
                }
            }

            GPU_SMJ::GPU_SMJ(uint64_t numDistinctKeys, uint64_t batchSize,
                             uint64_t maxTupleInWindow) : numDistinctKeys(numDistinctKeys), batchSize(batchSize),
                                                          maxTupleInWindow(maxTupleInWindow) {
                CUDA_CHECK(cudaMalloc(&d_leftTuples, maxTupleInWindow * sizeof(Tuple)));
                CUDA_CHECK(cudaMalloc(&d_rightTuples, maxTupleInWindow * sizeof(Tuple)));

                CUDA_CHECK(cudaMalloc(&d_sortedLeftTuples, maxTupleInWindow * sizeof(Tuple)));
                CUDA_CHECK(cudaMalloc(&d_sortedRightTuples, maxTupleInWindow * sizeof(Tuple)));

                CUDA_CHECK(cudaMalloc(&d_leftOccupation, numDistinctKeys * sizeof(unsigned long long)));
                CUDA_CHECK(cudaMalloc(&d_rightOccupation, numDistinctKeys * sizeof(unsigned long long)));

                CUDA_CHECK(cudaMalloc(&d_leftHistogram, numDistinctKeys * sizeof(unsigned long long int)));
                CUDA_CHECK(cudaMalloc(&d_rightHistogram, numDistinctKeys * sizeof(unsigned long long int)));

                CUDA_CHECK(cudaMalloc(&d_leftPrefixSum, numDistinctKeys * sizeof(uint64_t)));
                CUDA_CHECK(cudaMalloc(&d_rightPrefixSum, numDistinctKeys * sizeof(uint64_t)));

                CUDA_CHECK(cudaMalloc(&d_resultCount, sizeof(unsigned long long int)));
                h_resultCount = static_cast<unsigned long long int *>(std::malloc(sizeof(unsigned long long int)));

                CUDA_CHECK(cudaStreamCreate(&leftStream));
                CUDA_CHECK(cudaStreamCreate(&rightStream));
                CUDA_CHECK(cudaStreamCreate(&joinerStream));

            }

            GPU_SMJ::~GPU_SMJ() {
                CUDA_CHECK(cudaFree(d_leftTuples));
                CUDA_CHECK(cudaFree(d_rightTuples));

                CUDA_CHECK(cudaFree(d_sortedLeftTuples));
                CUDA_CHECK(cudaFree(d_sortedRightTuples));

                CUDA_CHECK(cudaFree(d_leftOccupation));
                CUDA_CHECK(cudaFree(d_rightOccupation));

                CUDA_CHECK(cudaFree(d_leftHistogram));
                CUDA_CHECK(cudaFree(d_rightHistogram));

                CUDA_CHECK(cudaFree(d_leftPrefixSum));
                CUDA_CHECK(cudaFree(d_rightPrefixSum));

                CUDA_CHECK(cudaFree(d_resultCount));
                free(h_resultCount);

                CUDA_CHECK(cudaStreamDestroy(leftStream));
                CUDA_CHECK(cudaStreamDestroy(rightStream));
                CUDA_CHECK(cudaStreamDestroy(joinerStream));
            }

            void GPU_SMJ::launchHistogramKernel(Tuple *buildSide, const uint64_t tupleCounts,
                                                unsigned long long int *histogram, cudaStream_t cudaStream) {
                dim3 dimBlock(1024, 1, 1); // vx * vy * vz must be <= 1024,
                dim3 dimGrid((tupleCounts + dimBlock.x - 1) / dimBlock.x, 1, 1);

                smjHistogramKernel<<<dimGrid, dimBlock, 0, cudaStream>>>(buildSide,
                                                                         tupleCounts,
                                                                         histogram);
            }

            void GPU_SMJ::launchSortKernel(Tuple *input, Tuple *output, uint64_t *prefixSum, uint64_t nTuples,
                                           unsigned long long int *occupation, cudaStream_t cudaStream) {
                dim3 dimBlock(32, 1, 1); // vx * vy * vz must be <= 1024,
                dim3 dimGrid((nTuples + dimBlock.x - 1) / dimBlock.x, 1, 1);
                CUDA_CHECK(cudaStreamSynchronize(cudaStream));
                smjSortKernel<<<dimGrid, dimBlock, 0, cudaStream>>>(input,
                                                                    output,
                                                                    prefixSum,
                                                                    nTuples,
                                                                    occupation
                );
            }

            void GPU_SMJ::launchMergeCountKernel(const unsigned long long int *leftHistogram,
                                                 const unsigned long long int *rightHistogram,
                                                 unsigned long long int *resultCount, uint64_t numDistinctKeys,
                                                 cudaStream_t cudaStream) {
                dim3 dimBlock(1024, 1, 1); // vx * vy * vz must be <= 1024,
                dim3 dimGrid((numDistinctKeys + dimBlock.x - 1) / dimBlock.x, 1, 1);
                smjMergeKernelCount<<<dimGrid, dimBlock, 0, cudaStream>>>(leftHistogram,
                                                                          rightHistogram,
                                                                          resultCount, numDistinctKeys);

            }

            void GPU_SMJ::launchMergeJoinKernel(const unsigned long long int *leftHistogram,
                                                const unsigned long long int *rightHistogram, Tuple *sortedLeftTuples,
                                                Tuple *sortedRightTuples, unsigned long long int *resultCount,
                                                uint64_t numDistinctKeys, ResultTuple *resultTuple,
                                                cudaStream_t cudaStream) {
                dim3 dimBlockMerge(32, 1, 1); // vx * vy * vz must be <= 1024,
                dim3 dimGridMerge((numDistinctKeys + dimBlockMerge.x - 1) / dimBlockMerge.x, 1, 1);
                smjMergeKernel_ES<<<dimGridMerge, dimBlockMerge, 0, cudaStream>>>(
                        leftHistogram, rightHistogram, sortedLeftTuples, sortedRightTuples,
                        resultCount, numDistinctKeys, resultTuple);
            }

            void GPU_SMJ::launchMergeWriteKernel(const unsigned long long int *leftHistogram,
                                                 const unsigned long long int *rightHistogram,
                                                 Tuple *sortedLeftTuples,
                                                 Tuple *sortedRightTuples,
                                                 unsigned long long int *resultCount,
                                                 uint64_t numDistinctKeys,
                                                 cudaStream_t cudaStream,
                                                 ResultTuple *sinkBuffer,
                                                 uint64_t sinkBufferSize) {
                dim3 dimBlock(1024, 1, 1); // vx * vy * vz must be <= 1024,
                dim3 dimGrid((numDistinctKeys + dimBlock.x - 1) / dimBlock.x, 1, 1);
                smjMergeKernelWrite<<<dimGrid, dimBlock, 0, cudaStream>>>(leftHistogram,
                                                                          rightHistogram,
                                                                          sortedLeftTuples,
                                                                          sortedRightTuples,
                                                                          resultCount, numDistinctKeys,
                                                                          sinkBuffer,
                                                                          sinkBufferSize);

            }

            void GPU_SMJ::copyTuplesToLocalStore(Tuple *localStore, Tuple *ringBuffer, std::vector<uint64_t> indexes,
                                                 uint64_t batchSize, cudaStream_t cudaStream) {
                int64_t batchIdx = 0;
                for (const auto &start: indexes) {
                    CUDA_CHECK(cudaMemcpyAsync(localStore + batchIdx * batchSize, ringBuffer + start,
                                               batchSize * sizeof(Tuple),
                                               cudaMemcpyHostToDevice, cudaStream));
                    batchIdx++;
                }
            }

        }
    }
}
