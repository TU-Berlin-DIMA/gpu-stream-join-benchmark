#ifndef STREAMJOINBENCHMARK_CPU_SMJ_HPP
#define STREAMJOINBENCHMARK_CPU_SMJ_HPP

#include <sjb/windowing/eager/executor/BaseEagerExecutor.hpp>
#include <mutex>

namespace Windowing {
    namespace Eager {
        namespace Executor {
            class CPU_SMJ : public BaseEagerExecutor {
            public:
                CPU_SMJ();

                ~CPU_SMJ();

                uint64_t execute(Tuple *tupleBuffer, bool isLeftSide) final;

                uint64_t findMatches(Tuple *tupleBuffer, std::vector<Tuple> currentSideSortedTuple, std::array<uint64_t, DISTINCT_KEYS> &currentSideHistogram);

                void clearStates() final;

            private:
                /**
                 * @brief comparator to sort EventTuple
                 */
                static bool compareEventTuple(const Tuple &tuple, const Tuple &other) {
                    return tuple.key < other.key;
                }

                void updateHistogram(Tuple *incomingTuple, std::array<uint64_t, DISTINCT_KEYS> &histogram);

                std::mutex leftMutex;
                std::mutex rightMutex;

                std::vector<Tuple> sortedLeftTuples;
                std::vector<Tuple> sortedRightTuples;

                std::array<uint64_t, DISTINCT_KEYS> leftHistogram{0};
                std::array<uint64_t, DISTINCT_KEYS> rightHistogram{0};
            };
        }
    }
}


#endif //STREAMJOINBENCHMARK_CPU_SMJ_HPP
