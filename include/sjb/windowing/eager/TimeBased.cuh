
#ifndef SJB_CUDA_EAGER_TIMEBASED_CUH
#define SJB_CUDA_EAGER_TIMEBASED_CUH

#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_unordered_map.h>
#include <sjb/windowing/BaseWindowing.cuh>
#include <sjb/windowing/eager/executor/CPU/CPU_NLJ.hpp>
#include <sjb/windowing/eager/executor/CPU/CPU_HJ.hpp>
#include <sjb/windowing/eager/executor/CPU/CPU_SMJ.hpp>
#include <sjb/windowing/eager/executor/GPU/GPU_NLJ.cuh>
#include <sjb/windowing/eager/executor/GPU/GPU_HJ.cuh>
#include <sjb/windowing/eager/executor/GPU/GPU_SMJ.cuh>

namespace Windowing {
    namespace Eager {
        class TimeBased : public BaseWindowing {
        public:
            explicit TimeBased(Sink &sink);

            ~TimeBased();

            void onIncomingLeft(Tuple *tupleBuffer) override;

            void onIncomingRight(Tuple *tupleBuffer) override;

        private:
            std::shared_ptr<Executor::JOINER_CLASS> joiner;

            std::atomic<uint64_t> watermark = {0};
            std::atomic<uint64_t> latestExecutionTs = {0};

            std::atomic<uint64_t> totalIncomingLeft {0};
            std::atomic<uint64_t> totalIncomingRight {0};


        };
    }
}

#endif //SJB_CUDA_EAGER_TIMEBASED_CUH
