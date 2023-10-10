#include <cassert>
#include <chrono>
#include <sjb/windowing/lazy/executor/CPU/CPU_HJ.hpp>
#include <sjb/utils/Logger.hpp>
#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <atomic>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>


#define xstr(a) str(a)
#define str(a) #a

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            CPU_HJ::CPU_HJ(uint64_t distinctKeys, uint64_t batchSize, uint64_t nProbeThreads, uint64_t maxTupleInWindow) : distinctKeys(
                    distinctKeys), batchSize(batchSize), nProbeThreads(nProbeThreads), maxTupleInWindow(maxTupleInWindow) {
                htStore = std::vector<Tuple>(maxTupleInWindow);
            }

            CPU_HJ::~CPU_HJ() {
            }

            ExecutionStatistic CPU_HJ::execute(Tuple *leftRingBuffer, Tuple *rightRingBuffer,
                                               std::vector<uint64_t> leftIndexesToJoin,
                                               std::vector<uint64_t> rightIndexesToJoin,
                                               Sink &sink) {
                std::vector<uint64_t> htOccupation(distinctKeys);
                std::vector<uint64_t> htInserted(distinctKeys);
                std::vector<uint64_t> htPrefixSum(distinctKeys);

                auto t0 = std::chrono::high_resolution_clock::now();
                // build

                for (unsigned long i: leftIndexesToJoin) {
                    computePrefixSum(i, leftRingBuffer, htOccupation, htPrefixSum);

                }
                for (unsigned long i: leftIndexesToJoin) {
                    build(i, leftRingBuffer, htStore,
                          htOccupation, htPrefixSum, htInserted);
                }

                auto t1 = std::chrono::high_resolution_clock::now();

                // probe
                std::atomic<uint64_t> matchCount = {0};
                tbb::task_arena CPUHJProbeArena(nProbeThreads);
                CPUHJProbeArena.execute([&]() {
                    tbb::parallel_for_each(rightIndexesToJoin.begin(), rightIndexesToJoin.end(),
                                           [&](uint64_t i) {
                                               auto result = probe(i,
                                                                   rightRingBuffer,
                                                                   htStore,
                                                                   htOccupation,
                                                                   htPrefixSum);
                                               matchCount.fetch_add(result);
                                           });
                });
                auto t2 = std::chrono::high_resolution_clock::now();

                auto buildTime = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                auto probeTime = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

                totalBuildTime += buildTime;
                totalProbeTime += probeTime;
                totalExecutedWindows++;

                ExecutionStatistic es = ExecutionStatistic();
                es.resultCount = matchCount;
                es.buildTime = buildTime;
                es.probeTime = probeTime;

                return es;
            }


            void CPU_HJ::build(uint64_t startIdx,
                               Tuple *ringBuffer,
                               std::vector<Tuple> &store,
                               std::vector<uint64_t> &occupation,
                               std::vector<uint64_t> &prefixSum,
                               std::vector<uint64_t> &inserted) {
                for (uint64_t i = startIdx; i < startIdx + batchSize; i++) {
                    auto currentHash = ringBuffer[i].key % distinctKeys;
                    LOG_DEBUG("key:%u, inserted[currentKey]:%lu, prefixSum[currentKey]:%lu", currentHash,
                              inserted[currentHash], prefixSum[currentHash]);
                    store[prefixSum[currentHash] + inserted[currentHash]].key = ringBuffer[i].key;
                    store[prefixSum[currentHash] + inserted[currentHash]].val = ringBuffer[i].val;
                    store[prefixSum[currentHash] + inserted[currentHash]].ts = ringBuffer[i].ts;
                    inserted[currentHash]++;
                }
            }

            uint64_t CPU_HJ::probe(uint64_t startIdx, Tuple *ringBuffer,
                                   std::vector<Tuple> &store,
                                   std::vector<uint64_t> &occupation,
                                   std::vector<uint64_t> &prefixSum) {
                uint64_t count = 0;
                uint64_t compareCount = 0;
                for (uint64_t i = startIdx; i < startIdx + batchSize; i++) {
                    auto currentHash = ringBuffer[i].key % distinctKeys;
                    auto currentOccupation = occupation[currentHash];
                    auto pS = prefixSum[currentHash];
                    for (uint64_t j = pS; j < pS + currentOccupation; j++) {
                        compareCount++;
                        if (currentHash == store[j].key) {
                            count++;
                        }
                    }
                }
                return count;
            }

            void CPU_HJ::computePrefixSum(uint64_t startIdx, Tuple *ringBuffer,
                                          std::vector<uint64_t> &occupation,
                                          std::vector<uint64_t> &prefixSum) {
                for (uint64_t i = startIdx; i < startIdx + batchSize; i++) {
                    // increment the occupation at ringbuffer[j].key
                    occupation[ringBuffer[i].key]++;
                }

                // compute the prefixSum
                uint64_t runningSum = 0;
                for (uint64_t i = 0; i < occupation.size(); i++) {
                    prefixSum[i] = runningSum;
                    runningSum += occupation[i];
                }

            }
        }
    }
}