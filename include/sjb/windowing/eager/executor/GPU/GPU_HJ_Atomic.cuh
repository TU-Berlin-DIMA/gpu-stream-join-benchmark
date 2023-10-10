#ifndef STREAMJOINBENCHMARK_EAGER_GPU_HJ_Atomic_CUH
#define STREAMJOINBENCHMARK_EAGER_GPU_HJ_Atomic_CUH
#include <sjb/windowing/eager/executor/BaseEagerExecutor.hpp>
#include <atomic>
#include <tbb/concurrent_queue.h>
#include <sjb/utils/ResultTuple.hpp>

#define CUDASTREAMCOUNT 1

namespace Windowing {
    namespace Eager {
        namespace Executor {
            class GPU_HJ_Atomic : public BaseEagerExecutor {
            public:
                GPU_HJ_Atomic(uint64_t numDistinctKeys, uint64_t windowSize, uint64_t batchSize);

                ~GPU_HJ_Atomic();

                uint64_t execute(Tuple *tupleBuffer, bool isLeftSide,uint64_t windowTupleCount,Sink &sink) final;

                void clearStates() final;

                void onEndOfStream() override;

            private:
                Tuple *d_leftHashTable;
                Tuple *d_rightHashTable;

                uint32_t *d_leftHashTableOccupation;
                uint32_t *d_rightHashTableOccupation;

                std::atomic<uint64_t> leftWindowSize {0};
                std::atomic<uint64_t> rightWindowSize {0};

                std::vector<cudaStream_t> streams;
                cudaStream_t memsetStream;

                uint64_t numDistinctKeys;
                uint64_t windowSize;
                uint64_t batchSize;


                /**
                 * @brief Insert tuples to the hash table
                 * @param tuples an array of tuples to be inserted
                 * @param hashTable a pointer to a hashtable in the GPU memory
                 * @param hashTableOccupation a pointer to an array storing the occupation of the hashtable
                 */
                void buildHashTable(Tuple *tuples, Tuple *hashTable, uint32_t *hashTableOccupation);

                /**
                 * @brief Probe tuples to a hashtable to find join matches
                 * @param tuples an array of tuples
                 * @param hashTable a pointer to a hashTable in the GPU memory to be probed
                 * @param hashTableOccupation a pointer to an array storing the occupation of the hashtable
                 */
                uint64_t probeHashTable(Tuple *tuples, Tuple *hashTable, uint32_t *hashTableOccupation,unsigned long long int *resultCount,ResultTuple *sinkBuffer,uint64_t sinkBufferSize);


		// stores count of join matches
		unsigned long long *d_matchCount;
		uint64_t *h_matchCount;
            };
        }
    }
}
#endif //STREAMJOINBENCHMARK_EAGER_GPU_HJ_Atomic_CUH
