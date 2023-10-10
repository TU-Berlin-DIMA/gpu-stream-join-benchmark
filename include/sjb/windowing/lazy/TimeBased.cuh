#ifndef STREAMJOINBENCHMARK_TIMEBASEDLAZYWINDOWING_CUH
#define STREAMJOINBENCHMARK_TIMEBASEDLAZYWINDOWING_CUH

#include <sjb/windowing/BaseWindowing.cuh>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_unordered_map.h>
#include <sjb/windowing/lazy/executor/CPU/CPU_NLJ.hpp>
#include <sjb/windowing/lazy/executor/CPU/CPU_HJ.hpp>
#include <sjb/windowing/lazy/executor/CPU/CPU_SMJ.hpp>
#include <sjb/windowing/lazy/executor/GPU/GPU_NLJ.cuh>
#include <sjb/windowing/lazy/executor/GPU/GPU_HJ.cuh>
#include <sjb/windowing/lazy/executor/GPU/GPU_SMJ.cuh>
#include <sjb/windowing/lazy/executor/GPU/GPU_SABER.cuh>

namespace Windowing {
    namespace Lazy {
        class TimeBased :  public BaseWindowing {
        public:
            explicit TimeBased(Sink &sink);

            ~TimeBased();

            void onIncomingLeft(Tuple *tupleBuffer) override;

            void onIncomingRight(Tuple *tupleBuffer) override;

            void execute();

            Tuple *leftRingBuffer{};
            Tuple *rightRingBuffer{};

            template<typename T>
            void updateMaximum(std::atomic<T>& maximum_value, T const& value) noexcept {
                T prev_value = maximum_value;
                while(prev_value < value &&
                      !maximum_value.compare_exchange_weak(prev_value, value))
                {}
            }
        private:
            tbb::concurrent_queue<uint64_t> availableLeftBuffer;
            tbb::concurrent_queue<uint64_t> availableRightBuffer;
            tbb::concurrent_queue<uint64_t> executableLeftBuffer;
            tbb::concurrent_queue<uint64_t> executableRightBuffer;

            tbb::concurrent_queue<std::shared_ptr<Executor::JOINER_CLASS>> joiners;

            std::atomic<uint64_t> nextLeftTs = {0};
            std::atomic<uint64_t> nextRightTs = {0};
            uint64_t nextWindowTs = WINDOW_SIZE;

            std::atomic<uint64_t> watermark = {0};
            std::atomic<uint64_t> latestExecutionTs = {0};
        };
    }
}


#endif //STREAMJOINBENCHMARK_TIMEBASEDLAZYWINDOWING_CUH
