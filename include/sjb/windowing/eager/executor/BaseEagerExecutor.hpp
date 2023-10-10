#ifndef STREAMJOINBENCHMARK_BASEEAGEREXECUTOR_HPP
#define STREAMJOINBENCHMARK_BASEEAGEREXECUTOR_HPP

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <sjb/utils/Tuple.hpp>
#include <sjb/sink/Sink.h>

namespace Windowing {
    namespace Eager {
        namespace Executor {
            class BaseEagerExecutor {
            public:
                // Eager execution
                virtual uint64_t execute(Tuple *tupleBuffer, bool isLeftSide, uint64_t windowTupleCount,Sink &sink) = 0;

                virtual void clearStates() = 0;

                virtual void onEndOfStream() = 0;

            private:
                bool endOfStream = false;
            };
        }
    }
}


#endif //STREAMJOINBENCHMARK_BASEEAGEREXECUTOR_HPP
