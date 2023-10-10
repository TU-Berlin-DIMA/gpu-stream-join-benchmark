#ifndef STREAMJOINBENCHMARK_CPU_HJ_HPP
#define STREAMJOINBENCHMARK_CPU_HJ_HPP

#include <sjb/windowing/eager/executor/BaseEagerExecutor.hpp>
#include <mutex>
#include <atomic>

namespace Windowing {
    namespace Eager {
        namespace Executor {
            class CPU_HJ : public BaseEagerExecutor {
            public:
                CPU_HJ(uint64_t maxTupleInWindow, uint64_t numDistinctKeys, uint64_t batchSize);

                ~CPU_HJ();

                uint64_t execute(Tuple *tupleBuffer, bool isLeftSide, uint64_t windowTupleCount,Sink &sink) override;

                void clearStates() final;

                void onEndOfStream() override;

            private:
                std::vector<Tuple> leftHashTable;
                std::vector<Tuple> rightHashTable;

                std::vector<std::atomic<uint64_t>> leftHashTableOccupation;
                std::vector<std::atomic<uint64_t>>  rightHashTableOccupation;

                void build(Tuple *tuples, std::vector<Tuple> &hashTable, std::vector<std::atomic<uint64_t>> &occupation);

                uint64_t probe(Tuple *tuples, std::vector<Tuple> &hashTable, std::vector<std::atomic<uint64_t>> &occupation);

                std::mutex buildMutex;

                uint64_t maxTupleInWindow;
                uint64_t numDistinctKeys;
                uint64_t batchSize;
            };
        }
    }
}
#endif //STREAMJOINBENCHMARK_CPU_HJ_HPP
