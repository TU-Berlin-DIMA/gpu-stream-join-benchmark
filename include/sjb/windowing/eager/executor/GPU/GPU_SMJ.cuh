#ifndef STREAMJOINBENCHMARK_GPU_SMJ_CUH
#define STREAMJOINBENCHMARK_GPU_SMJ_CUH

#include <sjb/windowing/eager/executor/BaseEagerExecutor.hpp>

namespace Windowing {
    namespace Eager {
        namespace Executor {
            class GPU_SMJ : public BaseEagerExecutor {
            public:
                GPU_SMJ();

                ~GPU_SMJ();

                uint64_t execute(Tuple *tuples, bool isLeftSide) final;

                void clearStates() final;

            private:
                Tuple *d_leftWindow;
                Tuple *d_rightWindow;

                unsigned long long int leftWindowOccupation{0};
                unsigned long long int rightWindowOccupation{0};

                // stores incoming tuples
                Tuple *d_tuples;
                // store the prefix sum
                uint32_t *d_prefixSum;
                // store the histogram
                uint32_t *d_histogram;
                // store the match count
                unsigned long long int *d_matchCount;

                uint32_t *d_leftWindowPrefixSums;
                uint32_t *d_rightWindowPrefixSums;

                void computePrefixSum(Tuple *d_tuples, uint64_t size, uint32_t *h_histogram);

                void merge(Tuple *d_incomingTuples, uint32_t *d_incomingTuplePrefixSum, Tuple *d_window,
                               uint32_t *d_windowPrefixSums, unsigned long long int windowOccupation);

                uint32_t *d_occupation;
                Tuple *d_tmp;

                // exp use pinned memory for hMatchCounter
                unsigned long long int *h_matchCount;

                cudaStream_t mergeStream;
                cudaStream_t sortStream;
                cudaStream_t sortedTuplesCopyStream;
                cudaStream_t prefixSumCopyStream;
                cudaStream_t prefixSumStream;
                cudaStream_t histogramStream;
                cudaStream_t matchCountMemcpyStream;
                cudaStream_t incomingTuplesMemcpyStream;

                cudaStream_t lWindowPrefixSumMemsetStream;
                cudaStream_t rWindowPrefixSumMemsetStream;
                cudaStream_t histogramMemsetStream;
                cudaStream_t matchCountMemsetStream;
            };
        }
    }
}
#endif //STREAMJOINBENCHMARK_GPU_SMJ_CUH
