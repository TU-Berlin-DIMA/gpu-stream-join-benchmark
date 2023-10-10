#include <windowing/eager/executor/CPU/CPU_NLJ.hpp>

namespace Windowing {
    namespace Eager {
        namespace Executor {

            CPU_NLJ::~CPU_NLJ() {

            }

            CPU_NLJ::CPU_NLJ() {

            }

            void CPU_NLJ::clearStates() {
                leftTuples.clear();
                rightTuples.clear();
            }

            uint64_t CPU_NLJ::execute(Tuple *tupleBuffer, bool isLeftSide) {
                uint64_t count = 0;

                if (isLeftSide) {
                    // update existing state
                    leftTuples.insert(leftTuples.end(), tupleBuffer, tupleBuffer + TUPLE_BUFFER_SIZE);

                    // find join matches
                    for (uint64_t i = 0; i < TUPLE_BUFFER_SIZE; i++) {
                        for (auto &tuple: rightTuples) {
                            count += rightTuples[i].key == tuple.key;
                        }
                    }
                } else {
                    // update existing state
                    rightTuples.insert(rightTuples.end(), tupleBuffer, tupleBuffer + TUPLE_BUFFER_SIZE);

                    // find join matches
                    for (uint64_t i = 0; i < TUPLE_BUFFER_SIZE; i++) {
                        for (auto &tuple: leftTuples) {
                            count += leftTuples[i].key == tuple.key;
                        }
                    }
                }

                return count;
            }
        }
    }
}
