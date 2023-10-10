#ifndef SJB_CUDA_BASEEXECUTOR_HPP
#define SJB_CUDA_BASEEXECUTOR_HPP

#include <sjb/utils/ExecutionStatistic.hpp>
#include <sjb/utils/Tuple.hpp>
#include <vector>

// Forward declare Sink
class Sink;

namespace Windowing {
    namespace Lazy {
        namespace Executor {
            class BaseLazyExecutor {
            public:
                explicit BaseLazyExecutor() {};
                /**
                 * Execute the stream join algorithm
                 * @param leftRingBuffer pointer to the left ring buffer
                 * @param rightRingBuffer pointer to the right ring buffer
                 * @param leftIndexesToJoin indexes of left side tuples belonging to the current window
                 * @param rightIndexesToJoin indexes of left side tuples belonging to the current window
                 * @param sink reference to the Sink, i.e., to write the execution statistics and potentially join results
                 * @return an execution statistics containing join measurements
                 */
                virtual ExecutionStatistic execute(Tuple *leftRingBuffer,
                                                   Tuple *rightRingBuffer,
                                                   std::vector<uint64_t> leftIndexesToJoin,
                                                   std::vector<uint64_t> rightIndexesToJoin,
                                                   Sink &sink) = 0;
            };
        }
    }
}


#endif //SJB_CUDA_BASEEXECUTOR_HPP
