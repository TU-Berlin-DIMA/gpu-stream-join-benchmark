#include <mutex>
#include <utils/ErrorChecking.cuh>
#include <windowing/eager/TimeBased.cuh>
#include <utils/Logger.hpp>
#include <vector>
#include <thread>

namespace Windowing {
    namespace Eager {
        template<typename T>
        void updateMaximum(std::atomic<T>& maximum_value, T const& value) noexcept
        {
            T prev_value = maximum_value;
            while(prev_value < value &&
                  !maximum_value.compare_exchange_weak(prev_value, value))
            {}
        }

        std::mutex eagerTimeBasedMutex;

        TimeBased::TimeBased(Sink &sink) : BaseWindowing(sink) {
            joiner = std::make_shared<Executor::JOINER_CLASS>();
        }

        void TimeBased::onIncomingLeft(Tuple *tupleBuffer) {
            // update the watermark with the ts of the latest tuple
            updateMaximum(watermark, tupleBuffer[TUPLE_BUFFER_SIZE-1].ts);
            LOG_DEBUG("Watermark: %lu", watermark.load());

            auto deltaTs = watermark - latestExecutionTs;
            if (deltaTs >= WINDOW_SIZE) {
                joiner->clearStates();
                latestExecutionTs = watermark.load();
            }

            auto matchCount = joiner->execute(tupleBuffer, true);
            totalIncomingLeft += TUPLE_BUFFER_SIZE;

            sink.addToCounter(matchCount, watermark);
            sink.addToPossibleMatches(TUPLE_BUFFER_SIZE * totalIncomingRight);
        }

        void TimeBased::onIncomingRight(Tuple *tupleBuffer) {
            // update the watermark with the ts of the latest tuple
            updateMaximum(watermark, tupleBuffer[TUPLE_BUFFER_SIZE-1].ts);
            LOG_DEBUG("Watermark: %lu", watermark.load());

            auto deltaTs = watermark - latestExecutionTs;
            if (deltaTs >= WINDOW_SIZE) {
                joiner->clearStates();
                latestExecutionTs = watermark.load();
            }

            auto matchCount = joiner->execute(tupleBuffer, false);

            totalIncomingRight += TUPLE_BUFFER_SIZE;

            sink.addToCounter(matchCount, watermark);
            sink.addToPossibleMatches(TUPLE_BUFFER_SIZE * totalIncomingLeft);
        }

        TimeBased::~TimeBased() {

        }
    }
}

