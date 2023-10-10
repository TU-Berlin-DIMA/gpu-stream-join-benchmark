#include <sjb/windowing/eager/executor/GPU/GPU_NLJ_BM.cuh>
#include <sjb/utils/ErrorChecking.cuh>
#include <sjb/utils/CudaUtils.cuh>
#include <sjb/utils/Logger.hpp>
#include <sjb/utils/Tuple.hpp>

namespace Windowing {
    namespace Eager {
        namespace Executor {
            uint64_t GPU_NLJ_BM::execute(Tuple *tupleBuffer, bool isLeftSide, uint64_t windowTupleCount, Sink &sink) {
                uint64_t count = 0;

                cudaStream_t cudaStream;
                cudaStreamCreate(&cudaStream);


                if (isLeftSide) {
                    leftOccupation = windowTupleCount;

                    // Prepare the bitmap
                    int nBitmapPerLeftTuple = ceil(rightOccupation / BitmapCapacity);
                    uint64_t numOfBitmaps = ceil(batchSize * nBitmapPerLeftTuple);
                    BitmapType *bitmaps;
                    LOG_DEBUG("numOfBitmaps: %lu", numOfBitmaps);
                    CUDA_CHECK(cudaMallocHost(&bitmaps,
                                              numOfBitmaps * sizeof(BitmapType))); // does not work with cudaMallocHost
                    CUDA_CHECK(cudaMemset(bitmaps, 0, numOfBitmaps));

                    // update existing state
                    CUDA_CHECK(cudaMemcpyAsync(d_leftTuples + windowTupleCount, tupleBuffer,
                                               batchSize * sizeof(Tuple), cudaMemcpyHostToDevice, cudaStream));

                    // find join matches
                    launchNestedLoopBitmapKernel(tupleBuffer, d_rightTuples, batchSize, rightOccupation,
                                                 cudaStream, sink);
                } else {
                    rightOccupation = windowTupleCount;


                    // update existing state
                    CUDA_CHECK(cudaMemcpyAsync(d_rightTuples + windowTupleCount, tupleBuffer,
                                               batchSize * sizeof(Tuple), cudaMemcpyHostToDevice, cudaStream));

                    // find join matches
                    launchNestedLoopBitmapKernel(d_leftTuples, tupleBuffer, leftOccupation, batchSize,
                                                 cudaStream, sink);
                }

                CUDA_CHECK(cudaStreamSynchronize(cudaStream));
                CUDA_CHECK(cudaStreamDestroy(cudaStream));

                return count;
            }

            __global__ void
            getJoinResultBitmapEager(const Tuple *leftTuples, const Tuple *rightTuples, const uint64_t nLeftTuples,
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

            void GPU_NLJ_BM::launchNestedLoopBitmapKernel(const Tuple *incoming, const Tuple *existing,
                                                          const uint64_t nIncoming, uint64_t nExisting,
                                                          cudaStream_t cudastream, Sink &sink) {
                // Prepare the bitmap
                int nBitmapPerIncomingTuple = ceil(nExisting / BitmapCapacity);
                uint64_t numOfBitmaps = ceil(nIncoming * nBitmapPerIncomingTuple);
                BitmapType *bitmaps;
                LOG_DEBUG("numOfBitmaps: %lu", numOfBitmaps);
                CUDA_CHECK(cudaMallocHost(&bitmaps,
                                          numOfBitmaps * sizeof(BitmapType))); // does not work with cudaMallocHost
                CUDA_CHECK(cudaMemset(bitmaps, 0, numOfBitmaps));

                dim3 dimBlock(32, 1, 1);
                dim3 dimGrid((nIncoming + dimBlock.x - 1) / dimBlock.x, 1, 1);

                getJoinResultBitmapEager<<<dimGrid, dimBlock, 0, cudastream>>>(incoming, existing, nIncoming,
                                                                               nExisting,
                                                                               bitmaps);
                CUDA_CHECK(cudaStreamSynchronize(cudastream));

                for (uint64_t incomingTupleIndex = 0; incomingTupleIndex < nIncoming; incomingTupleIndex++) {
                    for (uint64_t bitmapIndex = 0; bitmapIndex < nBitmapPerIncomingTuple; bitmapIndex++) {
                        auto offset = (incomingTupleIndex * nBitmapPerIncomingTuple) + bitmapIndex;

                        BitmapType bitVal = bitmaps[offset];

                        uint64_t iter = 0;
                        while (bitVal != 0) {
                            uint8_t bit = bitVal & 1;
                            if (bit == 1) {
//                                // Locate the original tuple in the ring buffer
//                                uint64_t lIdxInLeftRingBuffer =
//                                        leftIndexesToJoin[ceil(leftTupleIndex / batchSize)] +
//                                        (leftTupleIndex % batchSize);
//                                uint64_t rightTupleIndex = (bitmapIndex * BitmapCapacity) + iter;
//                                uint64_t rIdxInRightRingBuffer = rightIndexesToJoin[
//                                                                         ceil(rightTupleIndex / batchSize)] +
//                                                                 (rightTupleIndex % batchSize);
//
//                                // Sanity check
//                                if (leftRingBuffer[lIdxInLeftRingBuffer].key !=  rightRingBuffer[rIdxInRightRingBuffer].key) {
//                                    LOG_ERROR("Lidx:%lu, LKey: %u, RIdx: %lu RKey: %u", leftTupleIndex, leftRingBuffer[lIdxInLeftRingBuffer].key,
//                                              (bitmapIndex * BitmapCapacity) + iter,  rightRingBuffer[rIdxInRightRingBuffer].key);
//                                }
//
//                                // Propagate the result to the sink
//                                sink.incrementCounterAndStore(leftRingBuffer[lIdxInLeftRingBuffer].key,
//                                                              rightRingBuffer[rIdxInRightRingBuffer].key,
//                                                              leftRingBuffer[lIdxInLeftRingBuffer].val,
//                                                              rightRingBuffer[rIdxInRightRingBuffer].val,
//                                                              leftRingBuffer[lIdxInLeftRingBuffer].ts,
//                                                              rightRingBuffer[rIdxInRightRingBuffer].ts);
                                sink.incrementCounterAndStore(0,0,0,0,0,0);
                            }
                            bitVal >>= 1;
                            iter++;
                        }
                    }
                }
            }


            void GPU_NLJ_BM::clearStates() {
                CUDA_CHECK(cudaMemset(d_leftTuples, 0, windowSize * sizeof(Tuple)));
                CUDA_CHECK(cudaMemset(d_rightTuples, 0, windowSize * sizeof(Tuple)));

                leftOccupation.store(0);
                rightOccupation.store(0);
            }

            GPU_NLJ_BM::~GPU_NLJ_BM() {
                CUDA_CHECK(cudaFree(d_leftTuples));
                CUDA_CHECK(cudaFree(d_rightTuples));
            }

            GPU_NLJ_BM::GPU_NLJ_BM(uint64_t batchSize, uint64_t windowSize) : batchSize(batchSize),
                                                                              windowSize(windowSize),
                                                                              BaseEagerExecutor() {
                CUDA_CHECK(cudaMalloc(&d_leftTuples, windowSize * sizeof(Tuple)));
                CUDA_CHECK(cudaMalloc(&d_rightTuples, windowSize * sizeof(Tuple)));

                CUDA_CHECK(cudaMalloc(&d_counter, sizeof(unsigned long long)));
                CUDA_CHECK(cudaMallocHost(&h_counter, sizeof(unsigned long long)));

                CUDA_CHECK(cudaMemset(d_leftTuples, 0, windowSize * sizeof(Tuple)));
                CUDA_CHECK(cudaMemset(d_rightTuples, 0, windowSize * sizeof(Tuple)));
            }

            void GPU_NLJ_BM::onEndOfStream() {
                // no-op
            }
        }
    }
}
