#include <sjb/windowing/lazy/executor/GPU/GPU_HJ.cuh>
#include <sjb/utils/ErrorChecking.cuh>
#include <sjb/utils/CudaUtils.cuh>
#include <algorithm>

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            __global__ void computeHistogram(Tuple *buildSide,
                                             const uint64_t leftTupleCounts,
                                             unsigned long long int *histogram,
                                             uint64_t numDistinctKeys) {
                uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

                if (i < leftTupleCounts) {
                    auto currentTuple = buildSide[i];
                    auto hash = currentTuple.key % numDistinctKeys;
                    atomicAdd(&histogram[hash], 1);
                }
            }

            __global__ void buildHashTable(Tuple *buildSide,
                                           const uint64_t leftTupleCount,
                                           Tuple *hashTable,
                                           const uint64_t *prefixSum,
                                           unsigned long long *occupation,
                                           uint64_t numDistinctKeys,
                                           uint64_t maxTupleInWindow) {
                uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

                if (i < leftTupleCount) {
                    auto hash = buildSide[i].key % numDistinctKeys;

                    auto oldCount = atomicAdd(&occupation[hash], 1);
                    hashTable[prefixSum[hash] + oldCount].key = buildSide[i].key;
                    hashTable[prefixSum[hash] + oldCount].val = buildSide[i].val;
                    hashTable[prefixSum[hash] + oldCount].ts = buildSide[i].ts;
                }
            }

            __global__ void probeCount(Tuple *rightTuples,
                                       const uint64_t tupleCount,
                                       Tuple *hashTable,
                                       const unsigned long long *histogram,
                                       const uint64_t *prefixSum,
                                       unsigned long long *resultCount,
                                       uint64_t numDistinctKeys) {
                uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

                if (i < tupleCount) {
                    auto tupleToProbe = rightTuples[i];
                    auto hash = tupleToProbe.key % numDistinctKeys;

                    uint64_t localCount = 0;
                    auto start = prefixSum[hash];
                    auto end = start + histogram[hash];
                    for (uint64_t j = start; j < end; j++) {
                        localCount += (tupleToProbe.key == hashTable[j].key);
                    }
                    atomicAdd(resultCount, localCount);

                }
            }

            __global__ void probeWrite(Tuple *rightTuples,
                                       const uint64_t tupleCount,
                                       Tuple *hashTable,
                                       const unsigned long long *histogram,
                                       const uint64_t *prefixSum,
                                       unsigned long long *resultCount,
                                       uint64_t numDistinctKeys,
                                       ResultTuple *sinkBuffer,
                                       uint64_t sinkBufferSize) {
                uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

                if (i < tupleCount) {
                    auto tupleToProbe = rightTuples[i];
                    auto hash = tupleToProbe.key % numDistinctKeys;

                    auto start = prefixSum[hash];
                    auto end = start + histogram[hash];

                    auto localCount = 0;
                    for (uint64_t j = start; j < end; j++) {
                        localCount += tupleToProbe.key == hashTable[j].key;
                    }
                    auto oldVal = atomicAdd(resultCount, localCount);
                    localCount = 0;
                    for (uint64_t j = start; j < end; j++) {
                        if (tupleToProbe.key == hashTable[j].key) {
                            localCount++;
                            sinkBuffer[(oldVal+localCount) % sinkBufferSize].lkey = hashTable[j].key;
                            sinkBuffer[(oldVal+localCount) % sinkBufferSize].rkey = tupleToProbe.key;
                            sinkBuffer[(oldVal+localCount) % sinkBufferSize].lVal = hashTable[j].val;
                            sinkBuffer[(oldVal+localCount) % sinkBufferSize].rVal = tupleToProbe.val;
                            sinkBuffer[(oldVal+localCount) % sinkBufferSize].lTs = hashTable[j].ts;
                            sinkBuffer[(oldVal+localCount) % sinkBufferSize].rTs = tupleToProbe.ts;
                        }
                    }
                }
            }


            __global__ void probeJoin(Tuple *rightTuples,
                                      const uint64_t tupleCount,
                                      Tuple *hashTable,
                                      const unsigned long long *histogram,
                                      const uint64_t *prefixSum,
                                      unsigned long long *resultCount,
                                      uint64_t numDistinctKeys,
                                      ResultTuple *d_resultTuple) {
                uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

                if (i < tupleCount) {
                    auto tupleToProbe = rightTuples[i];
                    auto hash = tupleToProbe.key % numDistinctKeys;


                    for (uint64_t j = prefixSum[hash]; j < prefixSum[hash] + histogram[hash]; j++) {
                        const auto tupleInHt = hashTable[j];

                        if (tupleToProbe.key == tupleInHt.key) {
                            auto oldCount = atomicAdd(resultCount, 1);
                            d_resultTuple[oldCount].lkey = tupleInHt.key;
                            d_resultTuple[oldCount].rkey = tupleInHt.key;
                            d_resultTuple[oldCount].lVal = tupleInHt.val;
                            d_resultTuple[oldCount].rVal = tupleToProbe.val;
                            d_resultTuple[oldCount].lTs = tupleInHt.ts;
                            d_resultTuple[oldCount].rTs = tupleToProbe.ts;
                        }
                    }
                }
            }


            void GPU_HJ::launchComputeHistogramKernel(Tuple *buildSide,
                                                      const uint64_t nBuildSideTuples,
                                                      unsigned long long int *histogram,
                                                      uint64_t numDistinctKeys,
                                                      cudaStream_t cudaStream) {
                // 1D make sense here because in the kernel we only refer the  tuple by vx
                dim3 dimBlock(1024, 1, 1);
                dim3 dimGrid((nBuildSideTuples + dimBlock.x - 1) / dimBlock.x, 1, 1);

                computeHistogram<<<dimGrid, dimBlock, 0, cudaStream>>>(buildSide,
                                                                       nBuildSideTuples,
                                                                       histogram,
                                                                       numDistinctKeys);
            }

            void GPU_HJ::launchProbeJoinKernel(Tuple *rightTuples, const uint64_t nRightTuple, uint64_t nLeftTuple,
                                               Tuple *hashTable, const unsigned long long int *histogram,
                                               const uint64_t *prefixSum, unsigned long long int *resultCount,
                                               uint64_t numDistinctKeys, ResultTuple *resultTuple,
                                               cudaStream_t cudaStream) {
                dim3 dimBlock(1024, 1, 1); // vx * vy * vz must be <= 1024
                dim3 dimGrid((nRightTuple + dimBlock.x - 1) / dimBlock.x, 1, 1);

                probeJoin<<<dimGrid, dimBlock, 0, cudaStream>>>(rightTuples,
                                                                nLeftTuple,
                                                                hashTable,
                                                                histogram,
                                                                prefixSum,
                                                                resultCount,
                                                                numDistinctKeys,
                                                                resultTuple);

            }

            void GPU_HJ::launchProbeCountKernel(Tuple *rightTuples, const uint64_t nRightTuple, uint64_t nLeftTuple,
                                                Tuple *hashTable,
                                                const unsigned long long int *histogram, const uint64_t *prefixSum,
                                                unsigned long long int *resultCount, uint64_t numDistinctKeys,
                                                cudaStream_t cudaStream) {
                dim3 dimBlock(64, 1, 1); // vx * vy * vz must be <= 1024
                dim3 dimGrid((nRightTuple + dimBlock.x - 1) / dimBlock.x, 1, 1);

                probeCount<<<dimGrid, dimBlock, 0, cudaStream>>>(rightTuples,
                                                                 nLeftTuple,
                                                                 hashTable,
                                                                 histogram,
                                                                 prefixSum,
                                                                 resultCount,
                                                                 numDistinctKeys);

            }

            void GPU_HJ::launchProbeWriteKernel(Tuple *rightTuples, const uint64_t nRightTuple, uint64_t nLeftTuple,
                                                Tuple *hashTable,
                                                const unsigned long long int *histogram, const uint64_t *prefixSum,
                                                unsigned long long int *resultCount, uint64_t numDistinctKeys,
                                                cudaStream_t cudaStream, ResultTuple *sinkBuffer,
                                                uint64_t sinkBufferSize) {
                dim3 dimBlock(1024, 1, 1); // vx * vy * vz must be <= 1024
                dim3 dimGrid((nRightTuple + dimBlock.x - 1) / dimBlock.x, 1, 1);

                probeWrite<<<dimGrid, dimBlock, 0, cudaStream>>>(rightTuples,
                                                                 nLeftTuple,
                                                                 hashTable,
                                                                 histogram,
                                                                 prefixSum,
                                                                 resultCount,
                                                                 numDistinctKeys,
                                                                 sinkBuffer,
                                                                 sinkBufferSize
                );

            }

            void
            GPU_HJ::launchBuildHashTableKernel(Tuple *buildSideTuples, const uint64_t nLeftTuples, Tuple *hashTable,
                                               const uint64_t *prefixSum, unsigned long long *occupation,
                                               uint64_t numDistinctKeys, uint64_t maxTupleInWindow,
                                               cudaStream_t cudaStream) {
                dim3 dimBlock(1024, 1, 1);
                dim3 dimGrid((nLeftTuples + dimBlock.x - 1) / dimBlock.x, 1, 1);

                buildHashTable<<<dimGrid, dimBlock, 0, cudaStream>>>(buildSideTuples,
                                                                     nLeftTuples,
                                                                     hashTable, prefixSum,
                                                                     occupation,
                                                                     numDistinctKeys,
                                                                     maxTupleInWindow);
            }

            void GPU_HJ::copyTuplesToLocalStore(Tuple *localStore, Tuple *ringBuffer, std::vector<uint64_t> indexes,
                                                uint64_t batchSize, cudaStream_t cudaStream) {
//                uint64_t batchIdx = 0;
//                for (uint64_t i: indexes) {
//                    CUDA_CHECK(cudaMemcpyAsync(localStore + batchIdx * batchSize, ringBuffer + i,
//                                               batchSize * sizeof(Tuple),
//                                               cudaMemcpyHostToDevice, cudaStream));
//                    batchIdx++;
//                }
                // Assuming batches are next to each other
                // get the minimum index
                auto minIdx = *std::min_element(indexes.begin(), indexes.end());

                CUDA_CHECK(cudaMemcpyAsync(localStore, ringBuffer + minIdx,
                                           indexes.size() * batchSize * sizeof(Tuple),
                                           cudaMemcpyHostToDevice, cudaStream));

            }


            void GPU_HJ::copyTuplesToLocalStore(Tuple *localStore, Tuple *ringBuffer, std::vector<uint64_t> indexes,
                                                uint64_t batchSize, const std::vector<cudaStream_t> &cudaStreams) {


                uint64_t batchIdx = 0;
                for (uint64_t i: indexes) {
                    CUDA_CHECK(cudaMemcpyAsync(localStore + batchIdx * batchSize, ringBuffer + i,
                                               batchSize * sizeof(Tuple),
                                               cudaMemcpyHostToDevice, cudaStreams[batchIdx % cudaStreams.size()]));
                    batchIdx++;

                    if (batchIdx > 0 && batchIdx % cudaStreams.size() == 0) {
                        for (auto stream: cudaStreams) {
                            cudaStreamSynchronize(stream);
                        }
                    }
                }
            }


            GPU_HJ::GPU_HJ(uint64_t numDistinctKeys, uint64_t batchSize, uint64_t maxTupleInWindow) : numDistinctKeys(
                    numDistinctKeys), batchSize(batchSize), maxTupleInWindow(maxTupleInWindow) {
                CUDA_CHECK(cudaMallocHost(&h_histogram, numDistinctKeys * sizeof(unsigned long long)));
                CUDA_CHECK(cudaMalloc(&d_histogram, numDistinctKeys * sizeof(unsigned long long)));

                CUDA_CHECK(cudaMallocHost(&h_prefixSum, numDistinctKeys * sizeof(uint64_t)));
                CUDA_CHECK(cudaMalloc(&d_prefixSum, numDistinctKeys * sizeof(uint64_t)));

                CUDA_CHECK(cudaMalloc(&d_hashTable, maxTupleInWindow * sizeof(Tuple)));

                CUDA_CHECK(cudaMalloc(&d_occupation, numDistinctKeys * sizeof(unsigned long long)));

                CUDA_CHECK(cudaMalloc(&d_resultCount, sizeof(unsigned long long)));
                CUDA_CHECK(cudaMallocHost(&h_resultCount, sizeof(unsigned long long)));

                CUDA_CHECK(cudaMalloc(&d_leftTuples, maxTupleInWindow * sizeof(Tuple)));
                CUDA_CHECK(cudaMalloc(&d_rightTuples, maxTupleInWindow * sizeof(Tuple)));

                CUDA_CHECK(cudaStreamCreate(&leftStream));
                CUDA_CHECK(cudaStreamCreate(&rightStream));
                CUDA_CHECK(cudaStreamCreate(&joinerStream));
            }

            GPU_HJ::~GPU_HJ() {
                CUDA_CHECK(cudaFreeHost(h_histogram));
                CUDA_CHECK(cudaFree(d_histogram));

                CUDA_CHECK(cudaFreeHost(h_prefixSum));
                CUDA_CHECK(cudaFree(d_prefixSum));

                CUDA_CHECK(cudaFree(d_hashTable));
                CUDA_CHECK(cudaFree(d_occupation));

                CUDA_CHECK(cudaFreeHost(h_resultCount));
                CUDA_CHECK(cudaFree(d_resultCount));

                CUDA_CHECK(cudaFree(d_leftTuples));
                CUDA_CHECK(cudaFree(d_rightTuples));

                CUDA_CHECK(cudaStreamDestroy(leftStream));
                CUDA_CHECK(cudaStreamDestroy(rightStream));
                CUDA_CHECK(cudaStreamDestroy(joinerStream));
            }

        }
    }
}