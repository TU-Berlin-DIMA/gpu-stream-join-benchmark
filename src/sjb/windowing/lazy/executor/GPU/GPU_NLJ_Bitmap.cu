#include <sjb/windowing/lazy/executor/GPU/GPU_NLJ_Bitmap.cuh>
#include <sjb/utils/ErrorChecking.cuh>
#include <sjb/utils/Logger.hpp>
#include <sjb/sink/Sink.h>

namespace Windowing {
    namespace Lazy {
        namespace Executor {

            GPU_NLJ_Bitmap::~GPU_NLJ_Bitmap() = default;

            GPU_NLJ_Bitmap::GPU_NLJ_Bitmap(uint64_t batchSize, uint64_t maxTupleInWindow) : GPU_NLJ(batchSize,
                                                                                                    maxTupleInWindow) {

            }

            ExecutionStatistic GPU_NLJ_Bitmap::execute(Tuple *leftRingBuffer, Tuple *rightRingBuffer,
                                                       std::vector<uint64_t> leftIndexesToJoin,
                                                       std::vector<uint64_t> rightIndexesToJoin, Sink &sink) {
                uint64_t nLeftTuples = leftIndexesToJoin.size() * batchSize;
                uint64_t nRightTuples = rightIndexesToJoin.size() * batchSize;

                // Copy tuples to the GPU
                copyTuplesToLocalStore(d_leftTuples, leftRingBuffer, leftIndexesToJoin, batchSize, leftStream);
                copyTuplesToLocalStore(d_rightTuples, rightRingBuffer, rightIndexesToJoin, batchSize, rightStream);

                cudaStreamSynchronize(leftStream);
                cudaStreamSynchronize(rightStream);

                // Reset the result counter to 0
                CUDA_CHECK(cudaMemsetAsync(d_resultCount, 0, sizeof(unsigned long long), joinerStream));
                CUDA_CHECK(cudaStreamSynchronize(joinerStream));

                // Prepare the bitmap
                int nBitmapPerLeftTuple = ceil(nRightTuples / BitmapCapacity);
                uint64_t numOfBitmaps = ceil(nLeftTuples * nBitmapPerLeftTuple);
                BitmapType *bitmaps;
                LOG_DEBUG("numOfBitmaps: %lu", numOfBitmaps);
                CUDA_CHECK(cudaMallocHost(&bitmaps,
                                          numOfBitmaps * sizeof(BitmapType))); // does not work with cudaMallocHost
                CUDA_CHECK(cudaMemset(bitmaps, 0, numOfBitmaps));

                // Launch the bitmap kernel
                launchNestedLoopBitmapKernel(d_leftTuples, d_rightTuples, nLeftTuples, nRightTuples, bitmaps,
                                             joinerStream);
                cudaStreamSynchronize(joinerStream);

                // Interpret the bitmap
                // TODO per 11 APR: the interpretation should loop over the sparse array,
                //  and access the original tuple at that index, i.e., instead of scanning the whole input tuple
                for (uint64_t leftTupleIndex = 0; leftTupleIndex < nLeftTuples; leftTupleIndex++) {
                    for (uint64_t bitmapIndex = 0; bitmapIndex < nBitmapPerLeftTuple; bitmapIndex++) {
                        auto offset = (leftTupleIndex * nBitmapPerLeftTuple) + bitmapIndex;

                        BitmapType bitVal = bitmaps[offset];

                        uint64_t iter = 0;
                        while (bitVal != 0) {
                            uint8_t bit = bitVal & 1;
                            if (bit == 1) {
                                // Locate the original tuple in the ring buffer
                                uint64_t lIdxInLeftRingBuffer =
                                        leftIndexesToJoin[ceil(leftTupleIndex / batchSize)] +
                                        (leftTupleIndex % batchSize);
                                uint64_t rightTupleIndex = (bitmapIndex * BitmapCapacity) + iter;
                                uint64_t rIdxInRightRingBuffer = rightIndexesToJoin[
                                                                         ceil(rightTupleIndex / batchSize)] +
                                                                 (rightTupleIndex % batchSize);

                                // Sanity check
                                if (leftRingBuffer[lIdxInLeftRingBuffer].key !=  rightRingBuffer[rIdxInRightRingBuffer].key) {
                                    LOG_ERROR("Lidx:%lu, LKey: %u, RIdx: %lu RKey: %u", leftTupleIndex, leftRingBuffer[lIdxInLeftRingBuffer].key,
                                              (bitmapIndex * BitmapCapacity) + iter,  rightRingBuffer[rIdxInRightRingBuffer].key);
                                }

                                // Propagate the result to the sink
                                sink.incrementCounterAndStore(leftRingBuffer[lIdxInLeftRingBuffer].key,
                                                              rightRingBuffer[rIdxInRightRingBuffer].key,
                                                              leftRingBuffer[lIdxInLeftRingBuffer].val,
                                                              rightRingBuffer[rIdxInRightRingBuffer].val,
                                                              leftRingBuffer[lIdxInLeftRingBuffer].ts,
                                                              rightRingBuffer[rIdxInRightRingBuffer].ts);
                            }
                            bitVal >>= 1;
                            iter++;
                        }
                    }
                }

                cudaFreeHost(bitmaps);

                ExecutionStatistic es = ExecutionStatistic();
                return es;
            }
        }
    }
}