#include <sjb/windowing/lazy/executor/GPU/GPU_NLJ.cuh>
#include <sjb/utils/Logger.hpp>
#include <sjb/utils/ErrorChecking.cuh>
#include <sjb/utils/CudaUtils.cuh>
#include <sjb/utils/ResultTuple.hpp>
#include <sjb/sink/Sink.h>

#define SetBit(data, y)    (data |= (1 << y))   /* Set Data.Y to 1      */

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            GPU_NLJ::GPU_NLJ(uint64_t batchSize, uint64_t maxTupleInWindow)
                    : batchSize(batchSize),
                      maxTupleInWindow(maxTupleInWindow) {
                CUDA_CHECK(cudaMalloc(&d_resultCount, sizeof(unsigned long long)));
                CUDA_CHECK(cudaMallocHost(&h_resultCount, sizeof(unsigned long long)));

                cudaStreamCreate(&joinerStream);
                cudaStreamCreate(&leftStream);
                cudaStreamCreate(&rightStream);

                // Allocate GPU memory
                CUDA_CHECK(cudaMalloc(&d_leftTuples, maxTupleInWindow * sizeof(Tuple)));
                CUDA_CHECK(cudaMalloc(&d_rightTuples, maxTupleInWindow * sizeof(Tuple)));
            }

            __global__ void
            nestedLoopCountKernel(const Tuple *leftTuples, const Tuple *rightTuples, const uint64_t nLeftTuples,
                        uint64_t nRightTuples, unsigned long long *counter) {
                uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
                uint64_t j = blockIdx.y * blockDim.y + threadIdx.y;

                if (i < nLeftTuples && j < nRightTuples) {
                    auto &l = leftTuples[i];
                    auto &r = rightTuples[j];

                    if (l.key == r.key) {
                        auto oldCount = CudaUtils::atomicAggInc(counter);
                    }
                }
            }

            __global__ void
            nestedLoopJoinKernel(const Tuple *leftTuples, const Tuple *rightTuples, const uint64_t nLeftTuples,
                                  uint64_t nRightTuples, unsigned long long *counter, ResultTuple *d_resultTuple) {
                uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
                uint64_t j = blockIdx.y * blockDim.y + threadIdx.y;

                if (i < nLeftTuples && j < nRightTuples) {
                    auto &l = leftTuples[i];
                    auto &r = rightTuples[j];

                    if (l.key == r.key) {
                        auto oldCount = CudaUtils::atomicAggInc(counter);
                        d_resultTuple[oldCount].lkey = l.key;
                        d_resultTuple[oldCount].rkey = r.key;
                        d_resultTuple[oldCount].lVal = l.val;
                        d_resultTuple[oldCount].rVal = r.val;
                        d_resultTuple[oldCount].lTs = l.ts;
                        d_resultTuple[oldCount].rTs = r.ts;
                    }
                }
            }

            __global__ void
            getJoinResultBitmap(const Tuple *leftTuples, const Tuple *rightTuples, const uint64_t nLeftTuples,
                                uint64_t nRightTuples, BitmapType *bitmap) {
                uint64_t leftTupleIndex = blockIdx.x * blockDim.x + threadIdx.x;

                if (leftTupleIndex < nLeftTuples) {
                    auto bitmapIndex = leftTupleIndex * (nRightTuples / BitmapCapacity);
                    BitmapType localBitmap = 0;
                    int64_t shift = 0;
                    for (uint64_t rightTupleIndex = 0; rightTupleIndex < nRightTuples; rightTupleIndex++) {
                        if (leftTuples[leftTupleIndex].key == rightTuples[rightTupleIndex].key) {
                            SetBit(localBitmap, rightTupleIndex % BitmapCapacity);
                        }
                        if (((rightTupleIndex + 1) % BitmapCapacity) == 0) {
                            atomicAdd(&bitmap[bitmapIndex + shift], localBitmap);

                            localBitmap = 0; // reset/renew the local bitmap
                            shift++; // increment the shift
                        }
                    }
                }
            }


            void GPU_NLJ::launchNestedLoopCountKernel(const Tuple *leftTuples, const Tuple *rightTuples,
                                                      const uint64_t nLeftTuples, uint64_t nRightTuples,
                                                      unsigned long long int *counter, cudaStream_t cudaStream) {
                dim3 dimBlock(32, 32);
                dim3 dimGrid;
                dimGrid.x = (nLeftTuples + dimBlock.x - 1) / dimBlock.x;
                dimGrid.y = (nRightTuples + dimBlock.y - 1) / dimBlock.y;


                nestedLoopCountKernel<<<dimGrid, dimBlock, 0, cudaStream>>>(leftTuples, rightTuples, nLeftTuples, nRightTuples, counter);
            }

            void GPU_NLJ::launchNestedLoopJoinKernel(const Tuple *leftTuples, const Tuple *rightTuples,
                                                     const uint64_t nLeftTuples, uint64_t nRightTuples,
                                                     unsigned long long int *counter,  ResultTuple *d_resultTuple,
                                                     cudaStream_t cudaStream) {
                dim3 dimBlock(32, 32);
                dim3 dimGrid;
                dimGrid.x = (nLeftTuples + dimBlock.x - 1) / dimBlock.x;
                dimGrid.y = (nRightTuples + dimBlock.y - 1) / dimBlock.y;

                nestedLoopJoinKernel<<<dimGrid, dimBlock, 0, cudaStream>>>(leftTuples, rightTuples, nLeftTuples, nRightTuples, counter, d_resultTuple);
            }

            void GPU_NLJ::copyTuplesToLocalStore(Tuple *localStore, Tuple *ringBuffer, std::vector<uint64_t> indexes,
                                                 uint64_t batchSize, cudaStream_t cudaStream) {
                uint64_t batchIdx = 0;
                for (uint64_t i: indexes) {
                    CUDA_CHECK(cudaMemcpyAsync(localStore + batchIdx * batchSize, ringBuffer + i,
                                               batchSize * sizeof(Tuple),
                                               cudaMemcpyHostToDevice, cudaStream));
                    batchIdx++;
                }
            }

            GPU_NLJ::~GPU_NLJ() {
                CUDA_CHECK(cudaFreeHost(h_resultCount));
                CUDA_CHECK(cudaFree(d_resultCount));

                cudaStreamDestroy(joinerStream);
                cudaStreamDestroy(leftStream);
                cudaStreamDestroy(rightStream);

                cudaFree(d_leftTuples);
                cudaFree(d_rightTuples);
            }

            void GPU_NLJ::launchNestedLoopBitmapKernel(const Tuple *leftTuples, const Tuple *rightTuples,
                                                       const uint64_t nLeftTuples, uint64_t nRightTuples,
                                                       BitmapType *bitmaps, cudaStream_t cudastream) {
                dim3 dimBlock(32, 1, 1);
                dim3 dimGrid((nLeftTuples + dimBlock.x - 1) / dimBlock.x, 1, 1);

                getJoinResultBitmap<<<dimGrid, dimBlock, 0, cudastream>>>(leftTuples, rightTuples, nLeftTuples, nRightTuples,
                                                           bitmaps);
            }


        }
    }
}
