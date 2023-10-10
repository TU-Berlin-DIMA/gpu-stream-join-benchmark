#include <sjb/windowing/eager/executor/CPU/CPU_HJ.hpp>
#include <tbb/parallel_invoke.h>

namespace Windowing {
    namespace Eager {
        namespace Executor {

            CPU_HJ::CPU_HJ(uint64_t maxTupleInWindow, uint64_t numDistinctKeys, uint64_t batchSize) :
                    maxTupleInWindow(maxTupleInWindow),
                    numDistinctKeys(numDistinctKeys),
                    batchSize(batchSize) {
                leftHashTable.resize(maxTupleInWindow * numDistinctKeys);
                rightHashTable.resize(maxTupleInWindow * numDistinctKeys);

                leftHashTableOccupation = std::vector<std::atomic<uint64_t>>(numDistinctKeys);
                rightHashTableOccupation = std::vector<std::atomic<uint64_t>>(numDistinctKeys);
            }

            CPU_HJ::~CPU_HJ() {

            }

            void CPU_HJ::onEndOfStream() {
                // no-op
            }

            void CPU_HJ::clearStates() {
                for (uint64_t i = 0; i < numDistinctKeys; i++) {
                    leftHashTableOccupation[i].store(0);
                    rightHashTableOccupation[i].store(0);
                }

            }

            uint64_t CPU_HJ::execute(Tuple *tupleBuffer, bool isLeftSide, uint64_t windowTupleCount, Sink &sink) {
                uint64_t count = 0;

                if (isLeftSide) {
                    build(tupleBuffer, leftHashTable, leftHashTableOccupation);
                    count += probe(tupleBuffer, rightHashTable, rightHashTableOccupation);
                } else {
                    build(tupleBuffer, rightHashTable, rightHashTableOccupation);
                    count += probe(tupleBuffer, leftHashTable, leftHashTableOccupation);
                }


                return count;
            }

            void CPU_HJ::build(Tuple *tuples, std::vector<Tuple> &hashTable,
                               std::vector<std::atomic<uint64_t>> &occupation) {
                for (uint64_t i = 0; i < batchSize; i++) {
                    // Computing the hash
                    auto &currentTuple = tuples[i];
                    auto hash = currentTuple.key % numDistinctKeys;

                    // Increment the occupation
                    auto oldOccupation = occupation[hash].fetch_add(1);

                    // Updating the hash table's entry at the hash key
                    hashTable[hash * (maxTupleInWindow) + oldOccupation].key = currentTuple.key;
                    hashTable[hash * (maxTupleInWindow) + oldOccupation].val = currentTuple.val;
                    hashTable[hash * (maxTupleInWindow) + oldOccupation].ts = currentTuple.ts;
                }
            }

            uint64_t CPU_HJ::probe(Tuple *tuples, std::vector<Tuple> &hashTable,
                                   std::vector<std::atomic<uint64_t>> &occupation) {
                uint64_t probeMatchCount = 0;
                for (uint64_t i = 0; i < batchSize; i++) {
                    auto currentTuple = tuples[i];
                    auto hash = currentTuple.key % numDistinctKeys;

                    for (uint64_t j = 0; j < occupation[hash]; j++) {
                        auto tupleInBucket = hashTable[hash * (maxTupleInWindow) + j];
                        if (currentTuple.key == tupleInBucket.key) {
                            probeMatchCount++;
                        }
                    }
                }
                return probeMatchCount;
            }
        }
    }
}