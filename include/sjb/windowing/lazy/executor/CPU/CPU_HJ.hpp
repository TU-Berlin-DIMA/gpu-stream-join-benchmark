#ifndef SJB_CUDA_CPU_HJ_HPP
#define SJB_CUDA_CPU_HJ_HPP

#include <sjb/windowing/lazy/executor/BaseLazyExecutor.hpp>
#include <unordered_map>
#include <tbb/task_arena.h>
#include <tbb/parallel_for_each.h>

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            class CPU_HJ : public BaseLazyExecutor {
            public:
                CPU_HJ(uint64_t distinctKeys, uint64_t batchSize, uint64_t nProbeThreads, uint64_t maxTupleInWindow);

                virtual ~CPU_HJ();

                ExecutionStatistic execute(Tuple *leftRingBuffer,
                                           Tuple *rightRingBuffer,
                                           std::vector<uint64_t> leftIndexesToJoin,
                                           std::vector<uint64_t> rightIndexesToJoin,
                                           Sink &sink) final;

            private:
                void computePrefixSum(uint64_t startIdx,
                                      Tuple *ringBuffer,
                                      std::vector<uint64_t> &occupation,
                                      std::vector<uint64_t> &prefixSum);

                void build(uint64_t startIdx,
                           Tuple *ringBuffer,
                           std::vector<Tuple> &store,
                           std::vector<uint64_t> &occupation,
                           std::vector<uint64_t> &prefixSum,
                           std::vector<uint64_t> &inserted);

                uint64_t probe(uint64_t startIdx,
                               Tuple *ringBuffer,
                               std::vector<Tuple> &store,
                               std::vector<uint64_t> &occupation,
                               std::vector<uint64_t> &prefixSum);

                std::vector<Tuple> htStore;

                uint64_t totalBuildTime = 0;
                uint64_t totalProbeTime = 0;
                uint64_t totalExecutedWindows = 0;
                uint64_t distinctKeys;
                uint64_t batchSize;
                uint64_t nProbeThreads;
                uint64_t maxTupleInWindow;
            };
        }
    }
}

#endif //SJB_CUDA_CPU_HJ_HPP
