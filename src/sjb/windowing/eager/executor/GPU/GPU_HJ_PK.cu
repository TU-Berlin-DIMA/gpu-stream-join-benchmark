#include <sjb/utils/ErrorChecking.cuh>
#include <sjb/windowing/eager/executor/GPU/GPU_HJ_PK.cuh>
#include <sjb/utils/Logger.hpp>
#include <cassert>
#include <tbb/parallel_invoke.h>

namespace Windowing {
    namespace Eager {
        namespace Executor {
            __global__
            void persistentBuildKernel(volatile int8_t *signal,
                                       const volatile int8_t *keepAlive,
                                       volatile Tuple *workBuffer,
                                       Tuple *hashTable,
                                       uint32_t *hashTableOccupation,
                                       uint64_t batchSize,
                                       uint64_t windowSize,
                                       uint64_t numDistinctKeys) {
                auto index = blockIdx.x * blockDim.x + threadIdx.x;

                while (keepAlive[index] == 1) {
                    if (signal[index] == 1) {
                        if (index < batchSize) {
                            // Computing the hash
                            auto &currentTuple = workBuffer[index];
                            auto hash = currentTuple.key % numDistinctKeys;

                            // Increment the occupation
                            auto oldOccupation = atomicAdd(&hashTableOccupation[hash], 1);

                            // Updating the hash table's entry at the hash key
                            hashTable[hash * windowSize + oldOccupation].key = currentTuple.key;
                            hashTable[hash * windowSize + oldOccupation].val = currentTuple.val;
                            hashTable[hash * windowSize + oldOccupation].ts = currentTuple.ts;
                        }
                        signal[index] = 0;
                    }
                }
            }

            __global__
            void persistentProbeKernel(volatile int8_t *signal,
                                       const volatile int8_t *keepAlive,
                                       volatile Tuple *workBuffer,
                                       volatile Tuple *hashTable,
                                       const volatile uint32_t *hashTableOccupation,
                                       uint64_t batchSize,
                                       uint64_t windowSize,
                                       uint64_t numDistinctKeys,
                                       uint32_t *counter) {
                auto index = blockIdx.x * blockDim.x + threadIdx.x;

                while (keepAlive[index] == 1) {
                    if (signal[index] == 1) {
                        if (index < batchSize) {
                            auto hash = workBuffer[index].key % numDistinctKeys;
                            uint32_t localCounter = 0;
                            for (uint64_t j = 0; j < hashTableOccupation[hash]; j++) {
                                if (workBuffer[index].key == hashTable[hash * windowSize + j].key) {
                                    localCounter++;
                                }
                            }
                            counter[index] = localCounter;
                        }
                        signal[index] = 0;
                    }
                }
            }

            void GPU_HJ_PK::triggerKernelExecution(volatile int8_t *signals, uint32_t nThreadsToUse) {
                auto start = std::chrono::high_resolution_clock::now();

                // set the thread to ready-to-execute (by signalling 1)
                for (uint64_t i = 0; i < nThreadsToUse; i++) {
                    signals[i] = 1;
                }

                // busy loop waiting for kernel completion (i.e., by signalling 2)

                for (uint64_t i = 0; i < nThreadsToUse; i++) {
                    bool stopWaiting = false;
                    while (!stopWaiting) {
                        stopWaiting = signals[i] == 0;
                    }
                }

                auto end = std::chrono::high_resolution_clock::now();
//                printf("%lu\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
            }

            void printAverageOccupancy(uint32_t *hashTableOccupation, uint64_t count) {
                uint64_t sum =0;
                for (uint64_t i=0; i<count; i++) {
                    sum+=hashTableOccupation[i];
                }

                auto avgOccupancy = (double) sum / count;
//                printf("%.2f\n", avgOccupancy);
            }

            GPU_HJ_PK::GPU_HJ_PK(uint64_t numDistinctKeys, uint64_t windowSize, uint64_t batchSize) :
                    numDistinctKeys(numDistinctKeys),
                    windowSize(windowSize),
                    batchSize(batchSize) {
                // Currently, we simplify the case by assuming that we have enough persistent threads to process the whole batch
                assert(nPersistentThreads >= batchSize);

                // Allocate one hash table for each join side
                // In the worst case (e.g., skewed), all tuples belong to a single hash key. Thus, we need to prepare
                // as much as numDistinctKeys * windowSize
                CUDA_CHECK(cudaMallocHost(&leftHashTable, numDistinctKeys * windowSize * sizeof(Tuple)));
                CUDA_CHECK(cudaMallocHost(&rightHashTable, numDistinctKeys * windowSize * sizeof(Tuple)));

                for (uint64_t i = 0; i < numDistinctKeys * windowSize; i++) {
                    leftHashTable[i].key = 0;
                    leftHashTable[i].val = 0;
                    leftHashTable[i].ts = 0;

                    rightHashTable[i].key = 0;
                    rightHashTable[i].val = 0;
                    rightHashTable[i].ts = 0;
                }

                CUDA_CHECK(cudaMallocHost(&leftHashTableOccupation, numDistinctKeys * sizeof(uint32_t)));
                CUDA_CHECK(cudaMallocHost(&rightHashTableOccupation, numDistinctKeys * sizeof(uint32_t)));

                for (uint64_t i = 0; i < numDistinctKeys; i++) {
                    leftHashTableOccupation[i] = 0;
                    rightHashTableOccupation[i] = 0;
                }

                // Allocate work buffer for each join side
                // Each work buffer corresponds to each GPU kernel, i.e., to each workSignal, thus equal to numPersistentThreads
                CUDA_CHECK(cudaMallocHost(&leftWorkBuffer, nPersistentThreads * sizeof(Tuple)));
                CUDA_CHECK(cudaMallocHost(&rightWorkBuffer, nPersistentThreads * sizeof(Tuple)));

                // Allocate a global counter
                CUDA_CHECK(cudaMallocHost(&leftGlobalCounter, nPersistentThreads * sizeof(uint32_t)));
                CUDA_CHECK(cudaMallocHost(&rightGlobalCounter, nPersistentThreads * sizeof(uint32_t)));

                CUDA_CHECK(
                        cudaHostAlloc(&leftBuildWorkSignal, nPersistentThreads * sizeof(int8_t), cudaHostAllocMapped));
                CUDA_CHECK(cudaHostAlloc(&leftBuildKeepAliveSignal, nPersistentThreads * sizeof(int8_t),
                                         cudaHostAllocMapped));
                cudaStreamCreate(&leftBuildStream);
                instantiateKernel(leftBuildWorkSignal, leftBuildKeepAliveSignal, GPU_HJ_PK::LEFT_BUILD_KERNEL,
                                  leftBuildStream, leftWorkBuffer);

                CUDA_CHECK(
                        cudaHostAlloc(&leftProbeWorkSignal, nPersistentThreads * sizeof(int8_t), cudaHostAllocMapped));
                CUDA_CHECK(cudaHostAlloc(&leftProbeKeepAliveSignal, nPersistentThreads * sizeof(int8_t),
                                         cudaHostAllocMapped));
                cudaStreamCreate(&leftProbeStream);
                instantiateKernel(leftProbeWorkSignal, leftProbeKeepAliveSignal, GPU_HJ_PK::LEFT_PROBE_KERNEL,
                                  leftProbeStream, leftWorkBuffer);

                CUDA_CHECK(
                        cudaHostAlloc(&rightBuildWorkSignal, nPersistentThreads * sizeof(int8_t), cudaHostAllocMapped));
                CUDA_CHECK(cudaHostAlloc(&rightBuildKeepAliveSignal, nPersistentThreads * sizeof(int8_t),
                                         cudaHostAllocMapped));
                cudaStreamCreate(&rightBuildStream);
                instantiateKernel(rightBuildWorkSignal, rightBuildKeepAliveSignal, GPU_HJ_PK::RIGHT_BUILD_KERNEL,
                                  rightBuildStream, rightWorkBuffer);

                CUDA_CHECK(
                        cudaHostAlloc(&rightProbeWorkSignal, nPersistentThreads * sizeof(int8_t), cudaHostAllocMapped));
                CUDA_CHECK(cudaHostAlloc(&rightProbeKeepAliveSignal, nPersistentThreads * sizeof(int8_t),
                                         cudaHostAllocMapped));
                cudaStreamCreate(&rightProbeStream);
                instantiateKernel(rightProbeWorkSignal, rightProbeKeepAliveSignal, GPU_HJ_PK::RIGHT_PROBE_KERNEL,
                                  rightProbeStream, rightWorkBuffer);
            }

            uint64_t GPU_HJ_PK::execute(Tuple *tupleBuffer, bool isLeftSide) {
                auto start = std::chrono::high_resolution_clock::now();


                if (isLeftSide) {
                    for (uint64_t i = 0; i < batchSize; i++) {
                        leftGlobalCounter[i] = 0;
                    }

                    for (uint64_t i = 0; i < batchSize; i++) {
                        leftWorkBuffer[i].key = tupleBuffer[i].key;
                        leftWorkBuffer[i].val = tupleBuffer[i].val;
                        leftWorkBuffer[i].ts = tupleBuffer[i].ts;
                    }

                    tbb::parallel_invoke([&]() {
                        // build
                        triggerKernelExecution(leftBuildWorkSignal, batchSize);
                    }, [&]() {
                        // probe
                        triggerKernelExecution(leftProbeWorkSignal, batchSize);
                    });
                    printAverageOccupancy(leftHashTableOccupation, numDistinctKeys);

                    uint64_t total = 0;
                    for (uint64_t i = 0; i < batchSize; i++) {
                        total += leftGlobalCounter[i];
                    }

                    auto end = std::chrono::high_resolution_clock::now();
//                    printf("%lu\n", std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
                    return total;
                } else {
                    for (uint64_t i = 0; i < batchSize; i++) {
                        rightGlobalCounter[i] = 0;
                    }

                    for (uint64_t i = 0; i < batchSize; i++) {
                        rightWorkBuffer[i].key = tupleBuffer[i].key;
                        rightWorkBuffer[i].val = tupleBuffer[i].val;
                        rightWorkBuffer[i].ts = tupleBuffer[i].ts;
                    }

                    tbb::parallel_invoke([&]() {
                        // build
                        triggerKernelExecution(rightBuildWorkSignal, batchSize);
                    }, [&]() {
                        // probe
                        triggerKernelExecution(rightProbeWorkSignal, batchSize);
                    });
                    printAverageOccupancy(leftHashTableOccupation, numDistinctKeys);

                    uint64_t total = 0;
                    for (uint64_t i = 0; i < batchSize; i++) {
                        total += rightGlobalCounter[i];
                    }

                    auto end = std::chrono::high_resolution_clock::now();
//                    printf("%lu\n", std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
                    return total;
                }


            }

            void GPU_HJ_PK::onEndOfStream() {
                // Set the threads to terminate state (by signalling 0)
                for (uint64_t i = 0; i < nPersistentThreads; i++) {
                    leftBuildKeepAliveSignal[i] = 0;
                    leftProbeKeepAliveSignal[i] = 0;
                    rightBuildKeepAliveSignal[i] = 0;
                    rightProbeKeepAliveSignal[i] = 0;
                }

                CUDA_CHECK(cudaDeviceSynchronize());
            }

            void GPU_HJ_PK::clearStates() {
                auto start = std::chrono::high_resolution_clock::now();
                for (uint64_t i = 0; i < numDistinctKeys; i++) {
                    leftHashTableOccupation[i] = 0;
                    rightHashTableOccupation[i] = 0;
                }

                for (uint32_t i = 0; i < nPersistentThreads; i++) {
                    leftGlobalCounter[i] = 0;
                    rightGlobalCounter[i] = 0;
                }
                auto end = std::chrono::high_resolution_clock::now();
                printf("clearState time: %lu ms\n",
                       std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
            }

            GPU_HJ_PK::~GPU_HJ_PK() {
                cudaFreeHost(leftBuildWorkSignal);
                cudaFreeHost(leftBuildKeepAliveSignal);
                cudaFreeHost(leftProbeWorkSignal);
                cudaFreeHost(leftProbeKeepAliveSignal);
                cudaFreeHost(rightBuildWorkSignal);
                cudaFreeHost(rightBuildKeepAliveSignal);
                cudaFreeHost(rightProbeWorkSignal);
                cudaFreeHost(rightProbeKeepAliveSignal);

                cudaFreeHost(leftWorkBuffer);
                cudaFreeHost(rightWorkBuffer);

                cudaStreamDestroy(leftBuildStream);
                cudaStreamDestroy(leftProbeStream);
                cudaStreamDestroy(rightBuildStream);
                cudaStreamDestroy(rightProbeStream);
            }

            void GPU_HJ_PK::instantiateKernel(volatile int8_t *workSignal, volatile int8_t *keepAliveSignal,
                                              uint8_t kernelIdentifier, cudaStream_t stream, Tuple *workBuffer) const {

                for (uint64_t i = 0; i < nPersistentThreads; i++) {
                    workSignal[i] = 0;
                    keepAliveSignal[i] = 1;
                }

                // start the persistent kernel
                dim3 dimBlock(1024, 1, 1);
                dim3 dimGrid((nPersistentThreads + dimBlock.x - 1) / dimBlock.x, 1, 1);

                switch (kernelIdentifier) {
                    case GPU_HJ_PK::KernelIdentifier::LEFT_BUILD_KERNEL:
                        persistentBuildKernel<<<dimGrid, dimBlock, 0, stream>>>(workSignal, keepAliveSignal,
                                                                                leftWorkBuffer, leftHashTable,
                                                                                leftHashTableOccupation, batchSize,
                                                                                windowSize, numDistinctKeys);
                        break;
                    case GPU_HJ_PK::KernelIdentifier::LEFT_PROBE_KERNEL:
                        persistentProbeKernel<<<dimGrid, dimBlock, 0, stream>>>(workSignal,
                                                                                keepAliveSignal,
                                                                                workBuffer,
                                                                                rightHashTable,
                                                                                rightHashTableOccupation,
                                                                                batchSize,
                                                                                windowSize,
                                                                                numDistinctKeys,
                                                                                leftGlobalCounter);
                        break;
                    case GPU_HJ_PK::KernelIdentifier::RIGHT_BUILD_KERNEL:
                        persistentBuildKernel<<<dimGrid, dimBlock, 0, stream>>>(workSignal, keepAliveSignal,
                                                                                rightWorkBuffer, rightHashTable,
                                                                                rightHashTableOccupation, batchSize,
                                                                                windowSize, numDistinctKeys);
                        break;
                    case GPU_HJ_PK::KernelIdentifier::RIGHT_PROBE_KERNEL:
                        persistentProbeKernel<<<dimGrid, dimBlock, 0, stream>>>(workSignal,
                                                                                keepAliveSignal,
                                                                                workBuffer,
                                                                                leftHashTable,
                                                                                leftHashTableOccupation,
                                                                                batchSize,
                                                                                windowSize,
                                                                                numDistinctKeys,
                                                                                rightGlobalCounter);
                        break;
                    default:
                        throw std::runtime_error("Unknown kernel identifier.");
                }
            }
        }
    }
}