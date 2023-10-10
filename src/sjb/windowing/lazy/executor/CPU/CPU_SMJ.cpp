#include <algorithm>
#include <array>
#include <chrono>

#include <sjb/windowing/lazy/executor/CPU/CPU_SMJ.hpp>
#include <numeric>
#include <atomic>
#include <tbb/task_arena.h>
#include <tbb/parallel_sort.h>

#define xstr(a) str(a)
#define str(a) #a

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            CPU_SMJ::CPU_SMJ(uint64_t numDistinctKey, uint64_t batchSize, uint64_t nSortThreads) : numDistinctKey(
                    numDistinctKey),
                                                                                                   batchSize(batchSize),
                                                                                                   nSortThreads(
                                                                                                           nSortThreads) {}

            ExecutionStatistic
            CPU_SMJ::execute(Tuple *leftRingBuffer,
                             Tuple *rightRingBuffer,
                             std::vector<uint64_t> leftIndexesToJoin,
                             std::vector<uint64_t> rightIndexesToJoin,
                             Sink &sink) {
                auto t0 = std::chrono::high_resolution_clock::now();

                // compute the histogram & copy to local buffer
                auto leftHistogram = std::vector<uint64_t>(numDistinctKey);
                auto rightHistogram = std::vector<uint64_t>(numDistinctKey);

                auto leftLocalBuffer = std::vector<Tuple>(leftIndexesToJoin.size() * batchSize);
                auto rightLocalBuffer = std::vector<Tuple>(rightIndexesToJoin.size() * batchSize);

                uint64_t leftLocalBufferOccupation = 0;
                uint64_t rightLocalBufferOccupation = 0;

                for (const auto &start: leftIndexesToJoin) {
                    for (uint64_t i = start; i < start + batchSize; i++) {
                        auto &currentTuple = leftRingBuffer[i];

                        leftHistogram[currentTuple.key]++;

                        leftLocalBuffer[leftLocalBufferOccupation].key = currentTuple.key;
                        leftLocalBuffer[leftLocalBufferOccupation].val = currentTuple.key;
                        leftLocalBuffer[leftLocalBufferOccupation].ts = currentTuple.ts;
                        leftLocalBufferOccupation++;
                    }
                }

                for (const auto &start: rightIndexesToJoin) {
                    for (uint64_t i = start; i < start + batchSize; i++) {
                        auto &currentTuple = rightRingBuffer[i];

                        rightHistogram[currentTuple.key]++;

                        rightLocalBuffer[rightLocalBufferOccupation].key = currentTuple.key;
                        rightLocalBuffer[rightLocalBufferOccupation].val = currentTuple.val;
                        rightLocalBuffer[rightLocalBufferOccupation].ts = currentTuple.ts;
                        rightLocalBufferOccupation++;

                    }
                }

                auto t1 = std::chrono::high_resolution_clock::now();

                // compute prefix sum
                auto leftprefixSum = std::vector<uint64_t>(numDistinctKey);
                uint64_t leftRunningSum = 0;
                for (uint64_t leftKeyIdx = 0; leftKeyIdx < numDistinctKey; leftKeyIdx++) {
                    leftprefixSum[leftKeyIdx] = leftRunningSum;
                    leftRunningSum += leftHistogram[leftKeyIdx];
                }

                auto rightPrefixSum = std::vector<uint64_t>(numDistinctKey);
                uint64_t rightRunningSum = 0;
                for (uint64_t rightKeyIdx = 0; rightKeyIdx < numDistinctKey; rightKeyIdx++) {
                    rightPrefixSum[rightKeyIdx] = rightRunningSum;
                    rightRunningSum += rightHistogram[rightKeyIdx];
                }

                auto t2 = std::chrono::high_resolution_clock::now();

                // sort both relation
                tbb::task_arena CPUSMJSortArena(nSortThreads, 1, tbb::task_arena::priority::high);
                CPUSMJSortArena.execute([&] {
                    tbb::parallel_sort(leftLocalBuffer.begin(), leftLocalBuffer.end(), compareEventTuple);
                });
                CPUSMJSortArena.execute([&] {
                    tbb::parallel_sort(rightLocalBuffer.begin(), rightLocalBuffer.end(), compareEventTuple);
                });

                auto t3 = std::chrono::high_resolution_clock::now();

                // merge sorted vector
                auto distinctKeys = std::vector<uint64_t>(numDistinctKey);
                std::iota(distinctKeys.begin(), distinctKeys.end(), 0);

                // Merge method 1
                uint64_t count = 0;
                for (const auto keyIdx: distinctKeys) {
                    // edge case: if keyIdx is the last key
                    if (keyIdx == distinctKeys[numDistinctKey - 1]) {
                        // cartesian product
                        for (uint64_t i = leftprefixSum[keyIdx];
                             i < leftIndexesToJoin.size() * batchSize; i++) {
                            for (uint64_t j = rightPrefixSum[keyIdx];
                                 j < rightIndexesToJoin.size() * batchSize; j++) {
                                // match found
                                count++;
                            }
                        }
                    } else {
                        // genral case: cartesian product
                        for (uint64_t i = leftprefixSum[keyIdx]; i < leftprefixSum[keyIdx + 1]; i++) {
                            for (uint64_t j = rightPrefixSum[keyIdx]; j < rightPrefixSum[keyIdx + 1]; j++) {
                                // match found
                                count++;
                            }
                        }
                    }
                }

                auto t4 = std::chrono::high_resolution_clock::now();

                totalSortTime += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t0).count();
                totalMergeTime += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
                totalExecutedWindows++;

                ExecutionStatistic es = ExecutionStatistic();
                es.resultCount = count;
                es.sortTime = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t0).count();
                es.mergeTime = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

                return es;
            }

            CPU_SMJ::~CPU_SMJ() = default;


        }
    }
}


