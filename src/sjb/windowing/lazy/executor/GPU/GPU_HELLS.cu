#include <sjb/windowing/lazy/executor/GPU/GPU_HELLS.cuh>
#include <sjb/utils/ErrorChecking.cuh>
#include <sjb/utils/Logger.hpp>
#include <stdexcept>
#include <sjb/sink/Sink.h>

#define SetBit(data, y)    (data |= (1 << y))   /* Set Data.Y to 1      */

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            __global__ void
            getJoinResultBitmap(const Tuple *leftTuples, const Tuple *rightTuples, const uint64_t nLeftTuples,
                                uint64_t nRightTuples, BitmapType *bitmap) {
                uint64_t leftTupleIndex = blockIdx.x * blockDim.x + threadIdx.x;

                if (leftTupleIndex < nLeftTuples) {
                    auto bitmapIndex = leftTupleIndex * (nRightTuples / BitmapCapacity);
                    BitmapType localBitmap = 0;
                    int64_t shift = 0;
                    for (uint64_t rightTupleIndex = 0; rightTupleIndex < nRightTuples; rightTupleIndex++) {
//                        if (leftTupleIndex == 95 && rightTupleIndex == 127) {
//                            printf("localBitmap before::%u\n", localBitmap);
//                            printf("In-GPU LKey: %u, RKey:%u\n", leftTuples[leftTupleIndex].key,
//                                   rightTuples[rightTupleIndex].key);
//                        }
                        if (leftTuples[leftTupleIndex].key == rightTuples[rightTupleIndex].key) {
                            SetBit(localBitmap, rightTupleIndex % BitmapCapacity);
                        }
//                        if (leftTupleIndex == 95 && rightTupleIndex == 127) {
//                            printf("localBitmap after:%u\n", localBitmap);
//                        }
                        if (((rightTupleIndex + 1) % BitmapCapacity) == 0) {
                            atomicAdd(&bitmap[bitmapIndex + shift], localBitmap);

                            localBitmap = 0; // reset/renew the local bitmap
                            shift++; // increment the shift
                        }
                    }
                }
            }

            GPU_HELLS::GPU_HELLS(bool writeJoinResultToSink, uint64_t batchSize) : BaseLazyExecutor(
                    writeJoinResultToSink), batchSize(batchSize) {}

            ExecutionStatistic GPU_HELLS::execute(Tuple *leftRingBuffer, Tuple *rightRingBuffer,
                                                  std::vector<uint64_t> leftIndexesToCopy,
                                                  std::vector<uint64_t> rightIndexesToCopy,
                                                  Sink &sink) {
                uint64_t nLeftTuples = leftIndexesToCopy.size() * batchSize;
                uint64_t nRightTuples = rightIndexesToCopy.size() * batchSize;

                if (nRightTuples < BitmapCapacity) {
                    throw std::runtime_error("nRIghtTuples must be larger than BitmapCapacity");
                }

                // Prepare the bitmap
                int nBitmapPerLeftTuple = ceil(nRightTuples / BitmapCapacity);
                uint64_t numOfBitmaps = ceil(nLeftTuples * nBitmapPerLeftTuple);
                BitmapType *bitmaps;
                LOG_DEBUG("numOfBitmaps: %lu", numOfBitmaps);
                CUDA_CHECK(cudaMallocHost(&bitmaps, numOfBitmaps * sizeof(BitmapType))); // does not work with cudaMallocHost
                CUDA_CHECK(cudaMemset(bitmaps, 0, numOfBitmaps));
                cudaDeviceSynchronize();

                Tuple *leftTuples;
                Tuple *rightTuples;

                // Allocate GPU memory
                CUDA_CHECK(cudaMallocHost(&leftTuples, nLeftTuples * sizeof(Tuple)));
                CUDA_CHECK(cudaMallocHost(&rightTuples, nRightTuples * sizeof(Tuple)));

                // Copy tuples to the GPU
                for (uint64_t i = 0; i < leftIndexesToCopy.size(); i++) {
                    CUDA_CHECK(cudaMemcpy(leftTuples + i * batchSize, leftRingBuffer + leftIndexesToCopy[i],
                                          batchSize * sizeof(Tuple), cudaMemcpyHostToDevice));
                }

                for (uint64_t i = 0; i < rightIndexesToCopy.size(); i++) {
                    CUDA_CHECK(
                            cudaMemcpy(rightTuples + i * batchSize, rightRingBuffer + rightIndexesToCopy[i],
                                       batchSize * sizeof(Tuple), cudaMemcpyHostToDevice));
                }
                cudaDeviceSynchronize();

                LOG_DEBUG("Finished copying data");

                launchKernel(leftTuples, rightTuples, nLeftTuples,
                             nRightTuples, bitmaps);

                cudaDeviceSynchronize();
                ExecutionStatistic es = ExecutionStatistic();
                uint64_t resultCount = 0;
                for (uint64_t leftTupleIndex = 0; leftTupleIndex < nLeftTuples; leftTupleIndex++) {
                    for (uint64_t bitmapIndex = 0; bitmapIndex < nBitmapPerLeftTuple; bitmapIndex++) {
                        auto offset = (leftTupleIndex * nBitmapPerLeftTuple) + bitmapIndex;

                        BitmapType bitVal = bitmaps[offset];

                        uint64_t iter = 0;
                        while (bitVal != 0) {
                            uint8_t bit = bitVal & 1;
                            if (bit == 1) {
                                if (writeJoinResultToSink) {
                                    // TODO #78: propagate result back
                                    auto resultLKey = leftTuples[leftTupleIndex].key;
                                    auto resultRKey = rightTuples[(bitmapIndex * BitmapCapacity) + iter].key;

                                    if (resultLKey != resultRKey) {
                                        LOG_ERROR("Lidx:%lu, LKey: %u, RIdx: %lu RKey: %u", leftTupleIndex, resultLKey,
                                                  (bitmapIndex * BitmapCapacity) + iter, resultRKey);
                                    }

                                    // Propagate the result to the sink
                                    sink.incrementCounterAndStore(leftTuples[leftTupleIndex].key,
                                                                  leftTuples[leftTupleIndex].val,
                                                                  rightTuples[(bitmapIndex * BitmapCapacity) + iter].val,
                                                                  leftTuples[leftTupleIndex].ts,
                                                                  rightTuples[(bitmapIndex * BitmapCapacity) + iter].ts);
                                } else {
                                    resultCount++;
                                }
                            }
                            bitVal >>= 1;
                            iter++;
                        }
                    }
                }

                if (!writeJoinResultToSink) {
                    es.resultCount = resultCount;
                }

                cudaFreeHost(leftTuples);
                cudaFreeHost(rightTuples);
                cudaFreeHost(bitmaps);

                return es;
            }

            void GPU_HELLS::launchKernel(const Tuple *leftTuples, const Tuple *rightTuples,
                                         uint64_t nLeftTuples,
                                         uint64_t nRightTuples, BitmapType *bitmaps) {
                dim3 dimBlock(32, 1, 1);
                dim3 dimGrid((nLeftTuples + dimBlock.x - 1) / dimBlock.x, 1, 1);

                unsigned long long *d_counter;
                CUDA_CHECK(cudaMalloc(&d_counter, sizeof(unsigned long long)));
                CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(unsigned long long)));
                getJoinResultBitmap<<<dimGrid, dimBlock>>>(leftTuples, rightTuples, nLeftTuples, nRightTuples,
                                                           bitmaps);
            }

            GPU_HELLS::~GPU_HELLS() = default;
        }
    }
}
