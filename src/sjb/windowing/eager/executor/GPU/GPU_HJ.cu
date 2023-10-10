#include <sjb/utils/ErrorChecking.cuh>
#include <tbb/parallel_invoke.h>
#include <sjb/windowing/eager/executor/GPU/GPU_HJ.cuh>
#include <sjb/utils/Logger.hpp>

namespace Windowing {
    namespace Eager {
        namespace Executor {

            GPU_HJ::GPU_HJ(uint64_t numDistinctKeys, uint64_t windowSize, uint64_t batchSize) :
                    numDistinctKeys(numDistinctKeys),
                    windowSize(windowSize),
                    batchSize(batchSize) {
                // TODO: Use open addressing, as using windowSize will only safely work for count based windows
                CUDA_CHECK(cudaMalloc(&d_leftHashTable, numDistinctKeys * windowSize * sizeof(Tuple)));
                CUDA_CHECK(cudaMalloc(&d_rightHashTable, numDistinctKeys * windowSize * sizeof(Tuple)));

                CUDA_CHECK(cudaMalloc(&d_leftHashTableOccupation, numDistinctKeys * sizeof(uint32_t)));
                CUDA_CHECK(cudaMalloc(&d_rightHashTableOccupation, numDistinctKeys * sizeof(uint32_t)));

                CUDA_CHECK(cudaMallocHost(&h_matchCount, sizeof(uint64_t)));
                CUDA_CHECK(cudaMalloc(&d_matchCount, sizeof(unsigned long long)));
                CUDA_CHECK(cudaMemset(d_matchCount, 0, sizeof(unsigned long long)));

                for (uint32_t i = 0; i < CUDASTREAMCOUNT; i++) {
                    cudaStream_t cudaStream;
                    cudaStreamCreate(&cudaStream);
                    streams.push_back(cudaStream);
                }
                cudaStreamCreate(&memsetStream);
            }

            __global__ void buildEagerHashTable(Tuple *inputTuple, Tuple *hashTable, uint32_t *occupation,
                                                uint64_t batchSize, uint64_t numDistinctKeys, uint64_t windowSize) {
                uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

                if (i < batchSize) {
                    // Computing the hash
                    auto &currentTuple = inputTuple[i];
                    auto hash = currentTuple.key % numDistinctKeys;

                    // Increment the occupation
                    auto oldOccupation = atomicAdd(&occupation[hash], 1);

                    // Updating the hash table's entry at the hash key
                    hashTable[hash * windowSize + oldOccupation].key = currentTuple.key;
                    hashTable[hash * windowSize + oldOccupation].val = currentTuple.val;
                    hashTable[hash * windowSize + oldOccupation].ts = currentTuple.ts;
                }
            }

            __global__ void probeEager(const Tuple *tupleToProbe, const Tuple *hashTable,
                                       const uint32_t *hashTableOccupation, unsigned long long int *d_counter,
                                       uint64_t batchSize, uint64_t numDistinctKeys, uint64_t windowSize) {
                uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

                // TODO: What is the semantic of this?
                if (i == 1) {
                    (*d_counter) = 0;
                }

                if (i < batchSize) {
                    auto currentTuple = tupleToProbe[i];
                    auto hash = currentTuple.key % numDistinctKeys;

                    unsigned long long counter = 0;
                    for (uint64_t j = 0; j < hashTableOccupation[hash]; j++) {

                        if (currentTuple.key == hashTable[hash * windowSize + j].key) {
                            counter++;
                        }
                    }

                    atomicAdd(d_counter, counter);
                }
            }

            uint64_t GPU_HJ::execute(Tuple *tupleBuffer, bool isLeftSide,uint64_t windowTupleCount,Sink &sink) {
                uint64_t count = 0;
                if (isLeftSide) {
                    tbb::parallel_invoke([&]() {
                        // update existing state
                        buildHashTable(tupleBuffer, d_leftHashTable, d_leftHashTableOccupation);
                        leftWindowSize += batchSize;
                    }, [&]() {
                        // find join matches
                        if (rightWindowSize > 0) {
                            count += probeHashTable(tupleBuffer, d_rightHashTable, d_rightHashTableOccupation);
                        }
                    });

                } else {
                    tbb::parallel_invoke([&]() {
                        // update existing state
                        buildHashTable(tupleBuffer, d_rightHashTable, d_rightHashTableOccupation);
                        rightWindowSize += batchSize;
                    }, [&]() {
                        // find join matches
                        if (leftWindowSize > 0) {
                            count += probeHashTable(tupleBuffer, d_leftHashTable, d_leftHashTableOccupation);
                        }
                    });
                }

                return count;
            }


            void GPU_HJ::clearStates() {
                CUDA_CHECK(
                        cudaMemsetAsync(d_leftHashTableOccupation, 0, numDistinctKeys * sizeof(uint32_t),
                                        memsetStream));
                CUDA_CHECK(
                        cudaMemsetAsync(d_rightHashTableOccupation, 0, numDistinctKeys * sizeof(uint32_t),
                                        memsetStream));
                cudaStreamSynchronize(memsetStream);

                leftWindowSize = 0;
                rightWindowSize = 0;
            }

            void GPU_HJ::onEndOfStream() {
                // no-op
            }

            GPU_HJ::~GPU_HJ() {
                CUDA_CHECK(cudaFree(d_leftHashTable));
                CUDA_CHECK(cudaFree(d_rightHashTable));

                CUDA_CHECK(cudaFree(d_leftHashTableOccupation));
                CUDA_CHECK(cudaFree(d_rightHashTableOccupation));

                CUDA_CHECK(cudaFree(d_matchCount));
                CUDA_CHECK(cudaFreeHost(h_matchCount));

                for (uint32_t i = 0; i < CUDASTREAMCOUNT; i++) {
                    cudaStreamDestroy(streams[i]);
                }
                cudaStreamDestroy(memsetStream);
            }

            void GPU_HJ::buildHashTable(Tuple *tuples, Tuple *hashTable, uint32_t *hashTableOccupation) {
                auto &cudaStream = streams[tuples->key % CUDASTREAMCOUNT];

                // Build the hashtable from d_temporaryInputTuple
                dim3 dimBlock(1024, 1, 1);
                dim3 dimGrid((batchSize + dimBlock.x - 1) / dimBlock.x, 1, 1);

                buildEagerHashTable<<<dimGrid, dimBlock, 0, cudaStream>>>(
                        tuples,
                        hashTable,
                        hashTableOccupation,
                        batchSize,
                        numDistinctKeys,
                        windowSize);
                cudaStreamSynchronize(cudaStream);
            }

            uint64_t GPU_HJ::probeHashTable(Tuple *tuples, Tuple *hashTable, uint32_t *hashTableOccupation) {
                auto &cudaStream = streams[tuples->key % CUDASTREAMCOUNT];

                // Configure kernel execution parameters
                dim3 dimBlock(1024, 1, 1);
                dim3 dimGrid((batchSize + dimBlock.x - 1) / dimBlock.x, 1, 1);

                // Execute the kernel
                probeEager<<<dimGrid, dimBlock, 0, cudaStream>>>(
                        tuples,
                        hashTable,
                        hashTableOccupation,
                        d_matchCount,
                        batchSize,
                        numDistinctKeys,
                        windowSize);
                cudaStreamSynchronize(cudaStream);

                // Copy the match count
                CUDA_CHECK(cudaMemcpyAsync(h_matchCount, d_matchCount, sizeof(uint64_t), cudaMemcpyDeviceToHost,
                                           cudaStream));
                cudaStreamSynchronize(cudaStream);

                return *h_matchCount;
            }
        }
    }
}
