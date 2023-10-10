#include <sjb/windowing/lazy/executor/GPU/GPU_NLJ_NoOutput.cuh>
#include <sjb/utils/ErrorChecking.cuh>

namespace Windowing {
    namespace Lazy {
        namespace Executor {

            GPU_NLJ_NoOutput::~GPU_NLJ_NoOutput() {

            }

            GPU_NLJ_NoOutput::GPU_NLJ_NoOutput(uint64_t batchSize, uint64_t maxTupleInWindow) : GPU_NLJ(batchSize,
                                                                                                        maxTupleInWindow) {

            }

            ExecutionStatistic GPU_NLJ_NoOutput::execute(Tuple *leftRingBuffer, Tuple *rightRingBuffer,
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

                // Launch the count kernel
                launchNestedLoopCountKernel(d_leftTuples, d_rightTuples, nLeftTuples,
                                            nRightTuples, d_resultCount, joinerStream);
                cudaStreamSynchronize(joinerStream);

                // Copy the result count to the host
                CUDA_CHECK(cudaMemcpyAsync(h_resultCount, d_resultCount, sizeof(unsigned long long),
                                           cudaMemcpyDeviceToHost,
                                           joinerStream));
                cudaStreamSynchronize(joinerStream);


                ExecutionStatistic es = ExecutionStatistic();
                es.resultCount = *h_resultCount;

                return es;
            }
        }
    }
}