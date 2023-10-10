#include <windowing/lazy/TimeBased.cuh>
#include <utils/ErrorChecking.cuh>
#include <mutex>
#include <utils/Logger.hpp>
#include <thread>

namespace Windowing {
    namespace Lazy {
        template<typename T>
        void updateMaximum(std::atomic<T> &maximum_value, T const &value) noexcept {
            T prev_value = maximum_value;
            while (prev_value < value &&
                   !maximum_value.compare_exchange_weak(prev_value, value)) {}
        }

        std::mutex timeBasedWindowMutex;

        TimeBased::TimeBased(Sink &sink) : BaseWindowing(sink) {
            for (auto i = 0; i < RING_BUFFER_SIZE / TUPLE_BUFFER_SIZE; i += 1) {
                availableLeftBuffer.push(i * TUPLE_BUFFER_SIZE);
                availableRightBuffer.push(i * TUPLE_BUFFER_SIZE);
            }

            CUDA_CHECK(cudaMallocHost(&leftRingBuffer, RING_BUFFER_SIZE * sizeof(Tuple)));
            CUDA_CHECK(cudaMallocHost(&rightRingBuffer, RING_BUFFER_SIZE * sizeof(Tuple)));


            for (uint32_t i = 0; i < JOINER_THREADS; i++) {
                joiners.push(std::make_shared<Executor::JOINER_CLASS>());
            }
        }

        void TimeBased::onIncomingLeft(Tuple *tupleBuffer) {
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

                if (trial > RB_MAX_TRIES) {
                    LOG_ERROR("Ring buffer is full");
                    exit(1);
                }
            }

            // Write the incoming tuple to the next buffer
            std::memmove(leftRingBuffer + nextAvailableBuffer, tupleBuffer, TUPLE_BUFFER_SIZE * sizeof(Tuple));

            // update the watermark with the ts of the latest tuple
            updateMaximum(watermark, tupleBuffer[TUPLE_BUFFER_SIZE - 1].ts);
            LOG_DEBUG("Watermark: %lu", watermark.load());

            // Add the written buffer index to executableBuffer
            executableLeftBuffer.push(nextAvailableBuffer);

            // try to execute
            execute();
        }

        void TimeBased::onIncomingRight(Tuple *tupleBuffer) {
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

                if (trial > RB_MAX_TRIES) {
                    LOG_ERROR("Ring buffer is full");
                    exit(1);
                }
            }

            // write the incoming tuple to the next buffer
            std::memmove(rightRingBuffer + nextAvailableBuffer, tupleBuffer, TUPLE_BUFFER_SIZE * sizeof(Tuple));

            // update the watermark with the ts of the latest tuple
            updateMaximum(watermark, tupleBuffer[TUPLE_BUFFER_SIZE - 1].ts);
            LOG_DEBUG("Watermark: %lu", watermark.load());

            // add the written buffer index to executableBuffer
            executableRightBuffer.push(nextAvailableBuffer);

            // try to execute
            execute();
        }

        void TimeBased::execute() {

            timeBasedWindowMutex.lock();
            bool isWindowClosed = false;
            auto executableLeftSize = executableLeftBuffer.unsafe_size();
            auto executableRightSize = executableRightBuffer.unsafe_size();

            LOG_DEBUG("Available RB: %lu", availableRightBuffer.unsafe_size());

            auto deltaTs = watermark - latestExecutionTs;
            LOG_DEBUG("latestExecutionTs: %lu, Watermark %lu, Delta: %lu", latestExecutionTs.load(), watermark.load(),
                      deltaTs);
            if (deltaTs > (WINDOW_SIZE - 1)) {
                LOG_DEBUG("Window closed: latestExecutionTs: %lu, Watermark %lu, Delta: %lu", latestExecutionTs.load(),
                          watermark.load(), deltaTs);

                isWindowClosed = true;
                latestExecutionTs = watermark.load();
            }

            // check if we have enough buffer from both side
            std::vector<uint64_t> leftIndexesToJoin{};
            std::vector<uint64_t> rightIndexesToJoin{};

            if (isWindowClosed) {
                // copy the executable buffers to evaluable window
                for (auto i = 0; i < executableLeftSize; i++) {
                    uint64_t leftIdx;

                    executableLeftBuffer.try_pop(leftIdx);
                    leftIndexesToJoin.push_back(leftIdx);
                }

                for (auto i = 0; i < executableRightSize; i++) {
                    uint64_t rightIdx;

                    executableRightBuffer.try_pop(rightIdx);
                    rightIndexesToJoin.push_back(rightIdx);
                }
            }

            timeBasedWindowMutex.unlock();

            if (!leftIndexesToJoin.empty() && !rightIndexesToJoin.empty()) {
                std::shared_ptr<Executor::JOINER_CLASS> availableJoiner;
                bool isJoinerAvailable = false;
                while (!isJoinerAvailable) {
                    isJoinerAvailable = joiners.try_pop(availableJoiner);
                }

                auto executionStatistics = availableJoiner->execute(leftRingBuffer, rightRingBuffer, leftIndexesToJoin,
                                                                    rightIndexesToJoin);

                sink.addToCounter(executionStatistics.resultCount, watermark);
                sink.addToBuildTime(executionStatistics.buildTime);
                sink.addToProbeTime(executionStatistics.probeTime);
                sink.addToSortTime(executionStatistics.sortTime);
                sink.addToMergeTime(executionStatistics.mergeTime);
                sink.addToExecutedWindows(1);
                sink.addToPossibleMatches(
                        TUPLE_BUFFER_SIZE * leftIndexesToJoin.size() * TUPLE_BUFFER_SIZE * rightIndexesToJoin.size());

                LOG_DEBUG("#L:%lu, #R:%lu", leftIndexesToJoin.size() * TUPLE_BUFFER_SIZE,
                         rightIndexesToJoin.size() * TUPLE_BUFFER_SIZE);

                joiners.push(availableJoiner);
            }

            // put back left
            for (auto lIdx: leftIndexesToJoin) {
                availableLeftBuffer.push(lIdx);
            }

            // put back right
            for (auto rIdx: rightIndexesToJoin) {
                availableRightBuffer.push(rIdx);
            }
        }

        TimeBased::~TimeBased() {
            CUDA_CHECK(cudaFreeHost(leftRingBuffer));
            CUDA_CHECK(cudaFreeHost(rightRingBuffer));
        }

    }
}
