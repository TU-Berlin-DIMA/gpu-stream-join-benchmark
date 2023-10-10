#ifndef SJB_CUDA_CPU_NLJ_HPP
#define SJB_CUDA_CPU_NLJ_HPP

#include <tbb/concurrent_vector.h>
#include <sjb/windowing/lazy/executor/BaseLazyExecutor.hpp>

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            class CPU_NLJ : public BaseLazyExecutor {

            public:
                CPU_NLJ(uint64_t batchSize);

                ~CPU_NLJ();

                ExecutionStatistic execute(Tuple *leftRingBuffer,
                                           Tuple *rightRingBuffer,
                                           std::vector<uint64_t> leftIndexesToJoin,
                                           std::vector<uint64_t> rightIndexesToJoin,
                                           Sink &sink) final;

            private:
                tbb::concurrent_vector<Tuple> leftTuples{};
                tbb::concurrent_vector<Tuple> rightTuples{};
                uint64_t batchSize;

                void store(uint64_t startIdx, Tuple *ringBuffer, tbb::concurrent_vector<Tuple> &joinerState);

                uint64_t probe(uint64_t startIdx, Tuple *ringBuffer, tbb::concurrent_vector<Tuple> &joinerState);
            };
        }
    }
}


#endif //SJB_CUDA_CPU_NLJ_HPP