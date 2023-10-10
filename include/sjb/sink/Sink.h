#ifndef SJB_CUDA_SINK_H
#define SJB_CUDA_SINK_H

#include <atomic>
#include <sjb/utils/ExecutionStatistic.hpp>
#include <chrono>
#include <vector>
#include <tbb/concurrent_vector.h>
#include <sjb/utils/ResultTuple.hpp>
#include <array>
#include <cuda_runtime.h>

#define RT_KiB (1024/sizeof(ResultTuple))
#define RT_MiB (RT_KiB * 1024)
#define RT_GiB (RT_MiB * 1014)

struct Progress {
    uint64_t currentMatches;
    bool isWindowClosed;
    uint64_t timestamp;
};

class Sink {

public:
    Sink();

    void addToCounter(uint64_t currentMatches, bool isWindowClosed);

    void addToPossibleMatches(uint64_t count);

    uint64_t getMatchCount(bool useGPUCounter = false);

    const std::atomic<uint64_t> &getPossibleMatches() const;

    void addToBuildTime(uint64_t time);

    void addToProbeTime(uint64_t time);

    void addToSortTime(uint64_t time);

    void addToMergeTime(uint64_t time);

    void addToExecutedWindows(uint64_t time);

    uint64_t getSinkBufferSize() const;

    /**
     * @brief get average sort time in microseconds
     * @return average sort time
     */
    uint64_t getAverageSortTime();

    /**
     * @brief get average sort time in microseconds
     * @return average merge time
     */
    uint64_t getAverageMergeTime();

    /**
     * @brief get average sort time in microseconds
     * @return average build time
     */
    uint64_t getAverageBuildTime();

    /**
     * @brief get average sort time in microseconds
     * @return average probe time
     */
    uint64_t getAverageProbeTime();

    /**
     * Get the vector of progressiveness of the algorithm
     * @return the progressiveness concurrent vector of <time, number of result> pair
     */
    tbb::concurrent_vector<Progress> getProgress();

    void addToStatistics(ExecutionStatistic statistic);

    void markProcessingStart();

    void
    incrementCounterAndStore(uint32_t lkey, uint32_t rkey, uint32_t lVal, uint32_t rVal, uint64_t lTs, uint64_t rTs);

    void incrementCounterAndStore(ResultTuple *resultTuple, uint64_t tupleCount);

    const std::atomic<uint64_t> &getTotalExecutedWindows() const;

    virtual ~Sink();

    // Sink ring buffer to buffer result
    ResultTuple *sinkBuffer;
    uint64_t sinkBufferSize = 1 * RT_GiB; // store at max 4194304 result tuples
    cudaStream_t resultCopyStream;
    unsigned long long int *counterGPU;
private:
    void
    addToStore(uint64_t pos, uint32_t lkey, uint32_t rkey, uint32_t lVal, uint32_t rVal, uint64_t lTs, uint64_t rTs);

    // store the count  of result
    std::atomic<uint64_t> counter = {0};
    std::atomic<uint64_t> possibleMatches = {};

    // stores the build time, probe time of hash based join
    std::atomic<uint64_t> totalBuildTime = {0};
    std::atomic<uint64_t> totalProbeTime = {0};

    // stores the sort time and merge time of sort-merge join
    std::atomic<uint64_t> totalSortTime = {0};
    std::atomic<uint64_t> totalMergeTime = {0};

    // stores the number of executed windows
    std::atomic<uint64_t> totalExecutedWindows = {0};

    // store the timestamp creation of this sink
    std::chrono::time_point<std::chrono::high_resolution_clock> processingStartTime;
    // store the progress of the tested algorithm; each is a pair of <produced tuple, timestamp>
    tbb::concurrent_vector<Progress> progress;

    // store the latest result tuple
    // TODO: make this configurable, and should be large
    // TODO: this can also be GPU-accessible buffer
    std::array<ResultTuple, 1024> store;


};

#endif //SJB_CUDA_SINK_H
