#include <sjb/windowing/eager/executor/GPU/GPU_NLJ.cuh>
#include <sjb/utils/ErrorChecking.cuh>
#include <sjb/utils/CudaUtils.cuh>
#include <sjb/utils/Logger.hpp>
#include <sjb/utils/Tuple.hpp>

namespace Windowing {
    namespace Eager {
        namespace Executor {
            uint64_t GPU_NLJ::execute(Tuple *tupleBuffer, bool isLeftSide, uint64_t windowTupleCount,Sink &sink) {
                uint64_t count = 0;

                cudaStream_t cudaStream;
                cudaStreamCreate(&cudaStream);

                if (isLeftSide) {
                    // update existing state
                    CUDA_CHECK(cudaMemcpyAsync(d_leftTuples + windowTupleCount, tupleBuffer,
                                          batchSize * sizeof(Tuple), cudaMemcpyHostToDevice, cudaStream));
                    leftOccupation = windowTupleCount;

                    // find join matches
                    count += launchKernel(tupleBuffer, d_rightTuples, batchSize, rightOccupation,cudaStream);
                } else {
                    // update existing state
                    CUDA_CHECK(cudaMemcpyAsync(d_rightTuples + windowTupleCount, tupleBuffer,
                                          batchSize * sizeof(Tuple), cudaMemcpyHostToDevice, cudaStream));
                    rightOccupation = windowTupleCount;

                    // find join matches
                    count += launchKernel(d_leftTuples, tupleBuffer, leftOccupation, batchSize, cudaStream);
                }

                CUDA_CHECK(cudaStreamSynchronize(cudaStream));
                CUDA_CHECK(cudaStreamDestroy(cudaStream));

                return count;
            }

            __global__ void
            countKernel(const Tuple *leftTuples, const Tuple *rightTuples, const uint64_t nLeftTuples,
                        uint64_t nRightTuples,unsigned long long *counter) {
                uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
                uint64_t j = blockIdx.y * blockDim.y + threadIdx.y;

                if (i < nLeftTuples && j < nRightTuples) {
                    auto &l = leftTuples[i];
                    auto &r = rightTuples[j];

                    if (l.key == r.key) {
                        atomicAdd(counter, 1);
                    }
                }
            }

            uint64_t GPU_NLJ::launchKernel(Tuple *leftTuples, Tuple *rightTuples, uint64_t nLeftTuples,
                                           uint64_t nRightTuples, cudaStream_t cudaStream) {
                dim3 dimBlock(32, 32);
                dim3 dimGrid;
                dimGrid.x = (nLeftTuples + dimBlock.x - 1) / dimBlock.x;
                dimGrid.y = (nRightTuples + dimBlock.y - 1) / dimBlock.y;

                CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(unsigned long long),cudaStream));
                CUDA_CHECK(cudaStreamSynchronize(cudaStream));

                countKernel<<<dimGrid, dimBlock, 0, cudaStream>>>(leftTuples, rightTuples, nLeftTuples, nRightTuples,
                                                   d_counter);

                CUDA_CHECK(cudaMemcpyAsync(h_counter, d_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost, cudaStream));
                CUDA_CHECK(cudaStreamSynchronize(cudaStream));

                return *h_counter;
            }

            void GPU_NLJ::clearStates() {
                CUDA_CHECK(cudaMemset(d_leftTuples, 0, windowSize * sizeof(Tuple)));
                CUDA_CHECK(cudaMemset(d_rightTuples, 0, windowSize * sizeof(Tuple)));

//                leftOccupation.store(0);
//                rightOccupation.store(0);
            }

            GPU_NLJ::~GPU_NLJ() {
                CUDA_CHECK(cudaFree(d_leftTuples));
                CUDA_CHECK(cudaFree(d_rightTuples));
            }

            GPU_NLJ::GPU_NLJ(uint64_t batchSize, uint64_t windowSize) : batchSize(batchSize), windowSize(windowSize),
                                                                        BaseEagerExecutor() {
                CUDA_CHECK(cudaMalloc(&d_leftTuples, windowSize * sizeof(Tuple)));
                CUDA_CHECK(cudaMalloc(&d_rightTuples, windowSize * sizeof(Tuple)));

                CUDA_CHECK(cudaMalloc(&d_counter, sizeof(unsigned long long)));
                CUDA_CHECK(cudaMallocHost(&h_counter, sizeof(unsigned long long)));

                CUDA_CHECK(cudaMemset(d_leftTuples, 0, windowSize * sizeof(Tuple)));
                CUDA_CHECK(cudaMemset(d_rightTuples, 0, windowSize * sizeof(Tuple)));
            }

            void GPU_NLJ::onEndOfStream() {
                // no-op
            }
        }
    }
}
