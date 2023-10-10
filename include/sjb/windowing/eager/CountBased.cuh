
#ifndef SJB_CUDA_EAGERJOIN_COUNTBASED_CUH
#define SJB_CUDA_EAGERJOIN_COUNTBASED_CUH

#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_unordered_map.h>
#include <sjb/windowing/BaseWindowing.cuh>
#include <sjb/windowing/eager/executor/BaseEagerExecutor.hpp>
#include <sjb/windowing/eager/executor/GPU/GPU_HJ.cuh>
#include <sjb/windowing/eager/executor/GPU/GPU_SMJ.cuh>
#include <utility>

namespace Windowing {
    namespace Eager {
        std::mutex eagerMutex;

        class CountBased : public BaseWindowing {
        public:
            CountBased(Sink &sink,
                       uint64_t windowSize,
                       uint64_t batchSize,
                       std::shared_ptr<Executor::BaseEagerExecutor> joiner,
                       std::string algorithmName)
                    : BaseWindowing(sink, std::move(algorithmName)),
                      windowSize(windowSize),
                      batchSize(batchSize),
                      joiner(std::move(joiner)) {
             };

            const std::string &getAlgorithmName() const {
                return algorithmName;
            }

            ~CountBased() = default;

            void onIncomingLeft(Tuple *tupleBuffer) override {
                auto isWindowClosed = false;
                eagerMutex.lock();
                if (totalIncomingLeft >= windowSize || totalIncomingRight >= windowSize) {
                    joiner->clearStates();
                    totalIncomingLeft = 0;
                    totalIncomingRight = 0;
                    sink.addToExecutedWindows(1);
                    isWindowClosed = true;
                }
                eagerMutex.unlock();
                auto matchCount = joiner->execute(tupleBuffer, true, totalIncomingLeft, sink);

                totalIncomingLeft += batchSize;

                sink.addToCounter(matchCount, isWindowClosed);
                sink.addToPossibleMatches(batchSize * totalIncomingRight);
            };

            void onIncomingRight(Tuple *tupleBuffer) override {
                auto isWindowClosed = false;
                eagerMutex.lock();
                if (totalIncomingLeft >= windowSize || totalIncomingRight >= windowSize) {
                    joiner->clearStates();
                    totalIncomingLeft = 0;
                    totalIncomingRight = 0;
                    sink.addToExecutedWindows(1);
                    isWindowClosed = true;
                }
                eagerMutex.unlock();
                auto matchCount = joiner->execute(tupleBuffer, false, totalIncomingRight, sink);
                totalIncomingRight += batchSize;


                sink.addToCounter(matchCount, isWindowClosed);
                sink.addToPossibleMatches(batchSize * totalIncomingLeft);
            };

            void onEndOfStream() override {
                joiner->onEndOfStream();
            }

        private:
            std::atomic<uint64_t> totalIncomingLeft {0};
            std::atomic<uint64_t> totalIncomingRight {0};

            uint64_t windowSize;
            uint64_t batchSize;

            std::shared_ptr<Executor::BaseEagerExecutor> joiner;
        };


    }
}

#endif //SJB_CUDA_EAGERJOIN_COUNTBASED_CUH
