#include <windowing/eager/executor/CPU/CPU_SMJ.hpp>
#include <algorithm>

namespace Windowing {
    namespace Eager {
        namespace Executor {
            uint64_t CPU_SMJ::findMatches(Tuple *tupleBuffer, std::vector<Tuple> currentSideSortedTuple,
                                          std::array<uint64_t, DISTINCT_KEYS> &currentSideHistogram) {
                uint64_t count = 0;
//                auto incomingHistogram = std::array<uint64_t, DISTINCT_KEYS>();
//                for (uint64_t i = 0; i < TUPLE_BUFFER_SIZE; i++) {
//                    currentSideHistogram[tupleBuffer[i].key]++; <- this is not thread safe, in general, can we use exclusive scan here?
//                }
//                auto currentSidePrefixSum = std::array<uint64_t, DISTINCT_KEYS>();
//                uint64_t currentSideRunningSum = 0;
//                auto incomingPrefixSum = std::array<uint64_t, DISTINCT_KEYS>();
//                uint64_t incomingRunningSum = 0;
//                for (uint64_t key = 0; key < DISTINCT_KEYS; key++) {
//                    currentSidePrefixSum[key] = currentSideRunningSum;
//                    currentSideRunningSum += currentSideHistogram[key];
//                    incomingPrefixSum[key] = incomingRunningSum;
//                    incomingRunningSum += incomingHistogram[key];
//                }


//                auto distinctKeys = std::array<uint64_t, DISTINCT_KEYS>();
//                std::iota(distinctKeys.begin(), distinctKeys.end(), 0);
//

//                std::for_each(distinctKeys.begin(), distinctKeys.end(), [&](uint64_t keyIdx) {
//                    if (keyIdx == distinctKeys[DISTINCT_KEYS - 1]) {
//                        // cartesian product
//                        for (uint64_t i = currentSidePrefixSum[keyIdx]; i < currentSideSortedTuple.size(); i++) {
//                            for (uint64_t j = incomingPrefixSum[keyIdx]; j < TUPLE_BUFFER_SIZE; j++) {
//                                // match found
//                                count++;
//                            }
//                        }
//                    } else {
//                        // cartesian product
//                        for (uint64_t i = currentSidePrefixSum[keyIdx]; i < currentSidePrefixSum[keyIdx + 1]; i++) {
//                            for (uint64_t j = incomingPrefixSum[keyIdx]; j < incomingPrefixSum[keyIdx + 1]; j++) {
//                                // match found
//                                count++;
//                            }
//                        }
//                    }
//                });

                return count;
            }

            uint64_t CPU_SMJ::execute(Tuple *tupleBuffer, bool isLeftSide) {
                uint64_t count = 0;

                if (isLeftSide) {
                    // Insert
                    leftMutex.lock();
                    sortedLeftTuples.insert(sortedLeftTuples.end(), tupleBuffer, tupleBuffer + TUPLE_BUFFER_SIZE);
                    std::sort(sortedLeftTuples.begin(), sortedLeftTuples.end(), compareEventTuple);
                    leftMutex.unlock();

                    // Sort
                    std::sort(tupleBuffer, tupleBuffer + TUPLE_BUFFER_SIZE, compareEventTuple);

                    // Find matches
                    count = findMatches(tupleBuffer, sortedLeftTuples, leftHistogram);
                } else {
                    // Insert
                    rightMutex.lock();
                    sortedRightTuples.insert(sortedRightTuples.end(), tupleBuffer, tupleBuffer + TUPLE_BUFFER_SIZE);
                    std::sort(sortedRightTuples.begin(), sortedRightTuples.end(), compareEventTuple);
                    rightMutex.unlock();

                    // Sort
                    std::sort(tupleBuffer, tupleBuffer + TUPLE_BUFFER_SIZE, compareEventTuple);

                    // Find matches
                    count += findMatches(tupleBuffer, sortedRightTuples, rightHistogram);
                }
                return count;
            }

            void CPU_SMJ::clearStates() {
                sortedLeftTuples.clear();
                leftHistogram = std::array<uint64_t, DISTINCT_KEYS>{0};

                sortedRightTuples.clear();
                rightHistogram = std::array<uint64_t, DISTINCT_KEYS>{0};
            }

            void CPU_SMJ::updateHistogram(Tuple *incomingTuple, std::array<uint64_t, DISTINCT_KEYS> &histogram) {

            }

            CPU_SMJ::~CPU_SMJ() = default;

            CPU_SMJ::CPU_SMJ() = default;

        }
    }
}
