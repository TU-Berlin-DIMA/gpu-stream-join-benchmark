#ifndef SJB_CUDA_LAZYJOINER_CUH
#define SJB_CUDA_LAZYJOINER_CUH

#include <sjb/windowing/BaseWindowing.cuh>
#include <tbb/concurrent_queue.h>
#include <thread>
#include <sjb/utils/ErrorChecking.cuh>
#include <sjb/windowing/lazy/CountBased.cuh>
#include <mutex>
#include <sjb/utils/Logger.hpp>
#include <chrono>
#include <sjb/windowing/lazy/executor/BaseLazyExecutor.hpp>
#include <utility>

std::mutex executableBufferMutex;

namespace Windowing {
    namespace Lazy {
        class CountBased : public BaseWindowing {
        public:
            CountBased(Sink &sink, uint64_t ringBufferSize, uint64_t ringBufferMaxTries, uint64_t windowSize,
                       uint64_t batchSize, tbb::concurrent_queue<std::shared_ptr<Executor::BaseLazyExecutor>> &joiners,
                       std::string algorithmName)
                    : BaseWindowing(sink, std::move(algorithmName)),
                      ringBufferSize(ringBufferSize),
                      ringBufferMaxTries(ringBufferMaxTries),
                      windowSize(windowSize),
                      batchSize(batchSize),
                      joiners(joiners) {
                for (auto i = 0; i < ringBufferSize / batchSize; i += 1) {
                    availableLeftBuffer.push(i * batchSize);
                    availableRightBuffer.push(i * batchSize);
                }

                CUDA_CHECK(cudaMallocHost(&leftRingBuffer, ringBufferSize * sizeof(Tuple)));
                CUDA_CHECK(cudaMallocHost(&rightRingBuffer, ringBufferSize * sizeof(Tuple)));


            };

            const std::string &getAlgorithmName() const {
                return algorithmName;
            }

            void onIncomingLeft(Tuple *tupleBuffer) override {
                // Get the next available left buffer
                uint64_t nextAvailableBuffer;
                auto isPopSuccess = false;
                auto trial = 0;
                while (!isPopSuccess) {
                    isPopSuccess = availableLeftBuffer.try_pop(nextAvailableBuffer);

                    if (!isPopSuccess) {
                        std::this_thread::sleep_for(std::chrono::microseconds(1));
                        trial++;
                    }

                    if (trial > ringBufferMaxTries) {
                        LOG_ERROR("Ring buffer is full");
                        exit(1);
                    }
                }

                // Write the incoming tuple to the next buffer
                std::memmove(leftRingBuffer + nextAvailableBuffer, tupleBuffer, batchSize * sizeof(Tuple));
                // Add the written buffer index to executableBuffer
                executableLeftBuffer.push(nextAvailableBuffer);

                // try to execute
                execute();
            };

            void onIncomingRight(Tuple *tupleBuffer) override {
                // Get the next available right buffer
                uint64_t nextAvailableBuffer;

                auto isPopSuccess = false;
                auto trial = 0;
                while (!isPopSuccess) {
                    isPopSuccess = availableRightBuffer.try_pop(nextAvailableBuffer);

                    if (!isPopSuccess) {
                        std::this_thread::sleep_for(std::chrono::microseconds(1));
                        trial++;
                    }

                    if (trial > ringBufferMaxTries) {
                        LOG_ERROR("Ring buffer is full");
                        exit(1);
                    }
                }

                // write the incoming tuple to the next buffer
                std::memmove(rightRingBuffer + nextAvailableBuffer, tupleBuffer, batchSize * sizeof(Tuple));

                // add the written buffer index to executableBuffer
                executableRightBuffer.push(nextAvailableBuffer);

                // try to execute
                execute();
            };

            void execute() {
                // check if we have enough buffer from both side
                executableBufferMutex.lock();
                auto executableLeftSize = executableLeftBuffer.unsafe_size();
                auto executableRightSize = executableRightBuffer.unsafe_size();

                std::vector<uint64_t> leftIndexesToJoin{};
                std::vector<uint64_t> rightIndexesToJoin{};

                if (executableLeftSize * batchSize >= windowSize &&
                    executableRightSize * batchSize >= windowSize) {
                    // copy the executable buffers to evaluable window
                    for (auto i = 0; i < windowSize / batchSize; i++) {
                        uint64_t leftIdx;
                        uint64_t rightIdx;

                        executableLeftBuffer.try_pop(leftIdx);
                        leftIndexesToJoin.push_back(leftIdx);

                        executableRightBuffer.try_pop(rightIdx);
                        rightIndexesToJoin.push_back(rightIdx);
                    }
                }
                executableBufferMutex.unlock();

                if (!leftIndexesToJoin.empty() && !rightIndexesToJoin.empty()) {
                    std::shared_ptr<Executor::BaseLazyExecutor> availableJoiner;
                    bool isJoinerAvailable = false;
                    while (!isJoinerAvailable) {
                        isJoinerAvailable = joiners.try_pop(availableJoiner);
                    }

                    auto executionStatistics = availableJoiner->execute(leftRingBuffer, rightRingBuffer,
                                                                        leftIndexesToJoin,
                                                                        rightIndexesToJoin,
                                                                        sink);
                    LOG_DEBUG("Result count: %lu", executionStatistics.resultCount);

                    auto possibleMatches =
                            batchSize * leftIndexesToJoin.size() * batchSize *
                            rightIndexesToJoin.size();
                    sink.addToPossibleMatches(possibleMatches);

                    LOG_DEBUG("This Window's Selectivity: %.6f",
                              1.0 * executionStatistics.resultCount / possibleMatches);

                    LOG_DEBUG("Left Size:%lu , Right Size:%lu ", batchSize * leftIndexesToJoin.size(),
                              batchSize * rightIndexesToJoin.size());

                    sink.addToCounter(executionStatistics.resultCount, true);
                    sink.addToBuildTime(executionStatistics.buildTime);
                    sink.addToProbeTime(executionStatistics.probeTime);
                    sink.addToSortTime(executionStatistics.sortTime);
                    sink.addToMergeTime(executionStatistics.mergeTime);
                    sink.addToExecutedWindows(1);

                    for (uint32_t i = 0; i < windowSize / batchSize; i++) {
                        availableLeftBuffer.push(leftIndexesToJoin[i]);
                        availableRightBuffer.push(rightIndexesToJoin[i]);
                    }

                    joiners.push(availableJoiner);
                }
            };

            Tuple *leftRingBuffer{};
            Tuple *rightRingBuffer{};

            ~CountBased() {
                CUDA_CHECK(cudaFreeHost(leftRingBuffer));
                CUDA_CHECK(cudaFreeHost(rightRingBuffer));

                LOG_DEBUG("MemCpyInvoc: %lu", memcpyInvocationCount);
            }

            void onEndOfStream() override {
                // no-op
            }

        private:
            tbb::concurrent_queue<uint64_t> availableLeftBuffer;
            tbb::concurrent_queue<uint64_t> availableRightBuffer;
            tbb::concurrent_queue<uint64_t> executableLeftBuffer;
            tbb::concurrent_queue<uint64_t> executableRightBuffer;

            tbb::concurrent_queue<std::shared_ptr<Executor::BaseLazyExecutor>> joiners{};

            uint64_t memcpyInvocationCount = 0;

            uint64_t ringBufferSize;
            uint64_t ringBufferMaxTries;
            uint64_t windowSize;
            uint64_t batchSize;
        };
    }
}


#endif //SJB_CUDA_LAZYJOINER_CUH
