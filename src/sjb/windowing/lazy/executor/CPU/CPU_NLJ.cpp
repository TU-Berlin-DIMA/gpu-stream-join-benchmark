#include <sjb/windowing/lazy/executor/CPU/CPU_NLJ.hpp>
#include <sjb/utils/Logger.hpp>

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            CPU_NLJ::CPU_NLJ(uint64_t batchSize) : batchSize(batchSize) {}

            CPU_NLJ::~CPU_NLJ() = default;

            ExecutionStatistic CPU_NLJ::execute(Tuple *leftRingBuffer,
                                                Tuple *rightRingBuffer,
                                                std::vector<uint64_t> leftIndexesToJoin,
                                                std::vector<uint64_t> rightIndexesToJoin,
                                                Sink &sink) {

                uint64_t counter = 0;
                for (uint64_t i: leftIndexesToJoin) {
                    for (uint64_t j = i; j < i + batchSize; j++) {
                        for (uint64_t k: rightIndexesToJoin) {
                            for (uint64_t l = k; l < k + batchSize; l++) {
                                LOG_DEBUG("Left Key:%u, Right Key:%u", leftRingBuffer[j].key, rightRingBuffer[l].key);
                                counter += (leftRingBuffer[j].key == rightRingBuffer[l].key);
                            }
                        }
                    }
                }

                ExecutionStatistic es = ExecutionStatistic();
                es.resultCount = counter;
                return es;
            }

            void CPU_NLJ::store(uint64_t startIdx, Tuple *ringBuffer, tbb::concurrent_vector<Tuple> &joinerState) {
                for (uint64_t i = startIdx; i < startIdx + batchSize; i++) {
                    joinerState.push_back(Tuple{ringBuffer[i].key, ringBuffer[i].val, ringBuffer[i].ts});
                }
            }

            uint64_t CPU_NLJ::probe(uint64_t startIdx, Tuple *ringBuffer, tbb::concurrent_vector<Tuple> &joinerState) {
                uint64_t count = 0;
                for (uint64_t i = startIdx; i < startIdx + batchSize; i++) {
                    for (auto &tuple: joinerState) {
                        count += ringBuffer[i].key == tuple.key;
                    }
                }
                return count;
            }
        }
    }
}

