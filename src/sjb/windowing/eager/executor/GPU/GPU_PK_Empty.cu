#include <sjb/utils/ErrorChecking.cuh>
#include <sjb/windowing/eager/executor/GPU/GPU_PK_Empty.cuh>

namespace Windowing {
    namespace Eager {
        namespace Executor {
            __global__ void doWork(int i) {
                printf("executed from: %i\n", i);
            }

            __global__
            void persistent(volatile int8_t *signal, volatile int8_t *keepAlive, uint64_t batchSize,
                            cudaStream_t *cudaStreams) {
                int index = blockIdx.x * blockDim.x + threadIdx.x;


                while (keepAlive[index] == 1) {
                    if (signal[index] == 1) {
                        if (index < batchSize) {
                            for (int i = 0; i < batchSize; i++) {
                                doWork<<<1, 1>>>(i);
                            }
                        }
                        signal[index] = 0;
                    }
                }
            }

            void GPU_PK_Empty::triggerKernelExecution(volatile int8_t *signals, uint32_t nPersistentThreads) {
                // set the thread to ready-to-execute (by signalling 1)
                for (uint64_t i = 0; i < nPersistentThreads; i++) {
                    signals[i] = 1;
                }

                // busy loop waiting for kernel completion (i.e., by signalling 2)
                bool stopWaiting = false;
                while (!stopWaiting) {
                    for (uint64_t i = 0; i < nPersistentThreads; i++) {
                        if (signals[i] != 0) {
                            break; // keep waiting if the kernel is not done
                        } else if (i == nPersistentThreads - 1) { // we wait for the status of all threads
                            stopWaiting = true; // stop waiting if all threads have signal a done
                        }
                    }
                }
            }

            GPU_PK_Empty::GPU_PK_Empty(uint64_t numDistinctKeys, uint64_t windowSize, uint64_t batchSize) :
                    numDistinctKeys(numDistinctKeys),
                    windowSize(windowSize),
                    batchSize(batchSize) {
                CUDA_CHECK(cudaHostAlloc(&signals, nPersistentThreads * sizeof(int8_t), cudaHostAllocMapped));
                CUDA_CHECK(cudaHostAlloc(&keepAlive, nPersistentThreads * sizeof(int8_t), cudaHostAllocMapped));
                for (uint64_t i = 0; i < nPersistentThreads; i++) {
                    signals[i] = 0;
                    keepAlive[i] = 1;
                }

                CUDA_CHECK(cudaDeviceSynchronize());

                cudaStreams = static_cast<cudaStream_t *>(std::malloc(batchSize * sizeof(cudaStream_t)));
                for (uint64_t i = 0; i < batchSize; i++) {
                    cudaStreamCreate(&cudaStreams[i]);
                }

                // start the persistent kernel
//                dim3 dimBlock(1024, 1, 1);
//                dim3 dimGrid((nPersistentThreads + dimBlock.x - 1) / dimBlock.x, 1, 1);
//                persistent<<<dimGrid, dimBlock>>>(signals, keepAlive, batchSize);
                persistent<<<1, 1>>>(signals, keepAlive, batchSize, cudaStreams);
            }


            uint64_t GPU_PK_Empty::execute(Tuple *tupleBuffer, bool isLeftSide) {
                triggerKernelExecution(signals, nPersistentThreads);

                return 0;
            }

            void GPU_PK_Empty::onEndOfStream() {
                // Set the threads to terminate state (by signalling 0)
                for (uint64_t i = 0; i < nPersistentThreads; i++) {
                    keepAlive[i] = 0;
                }

                CUDA_CHECK(cudaDeviceSynchronize());
            }

            void GPU_PK_Empty::clearStates() {
                // no-op
            }

            GPU_PK_Empty::~GPU_PK_Empty() {
                cudaFreeHost(signals);

                for (uint64_t i = 0; i < batchSize; i++) {
                    cudaStreamDestroy(cudaStreams[i]);
                }
            }

        }
    }
}
