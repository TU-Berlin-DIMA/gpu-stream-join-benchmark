#ifndef SJB_CUDA_CPU_SMJ_HPP
#define SJB_CUDA_CPU_SMJ_HPP

#include <sjb/windowing/lazy/executor/BaseLazyExecutor.hpp>
#include <vector>
#include <numeric>

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            class CPU_SMJ : public BaseLazyExecutor{
            public:
                CPU_SMJ(uint64_t numDistinctKey, uint64_t batchSize, uint64_t nSortThreads);

                ~CPU_SMJ();

                ExecutionStatistic execute(Tuple *leftRingBuffer,
                                           Tuple *rightRingBuffer,
                                           std::vector<uint64_t> leftIndexesToJoin,
                                           std::vector<uint64_t> rightIndexesToJoin,
                                           Sink &sink) final;

            private:
                /**
                 * @brief comparator to sort EventTuple
                 */
                static bool compareEventTuple(const Tuple &tuple, const Tuple &other) {
                    return tuple.key < other.key;
                }

                uint64_t totalSortTime = 0;
                uint64_t totalMergeTime = 0;
                uint64_t totalExecutedWindows = 0;

                uint64_t numDistinctKey;
                uint64_t batchSize;

                uint64_t nSortThreads;
            };
        }
    }
}


#endif //SJB_CUDA_CPU_SMJ_HPP
