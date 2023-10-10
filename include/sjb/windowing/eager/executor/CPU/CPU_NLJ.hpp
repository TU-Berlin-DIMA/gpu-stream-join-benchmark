#ifndef STREAMJOINBENCHMARK_CPU_NLJ_HPP
#define STREAMJOINBENCHMARK_CPU_NLJ_HPP

#include <sjb/windowing/eager/executor/BaseEagerExecutor.hpp>

namespace Windowing {
    namespace Eager {
        namespace Executor {
            class CPU_NLJ : public BaseEagerExecutor {
            public:
                CPU_NLJ();

                ~CPU_NLJ();

                uint64_t execute(Tuple *tupleBuffer, bool isLeftSide);

                void clearStates() final;

            private:
                std::vector<Tuple> leftTuples;
                std::vector<Tuple> rightTuples;
            };
        }
    }
}

#endif //STREAMJOINBENCHMARK_CPU_NLJ_HPP
