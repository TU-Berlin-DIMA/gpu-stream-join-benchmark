#ifndef STREAMJOINBENCHMARK_EXECUTIONSTATISTICS_CUH
#define STREAMJOINBENCHMARK_EXECUTIONSTATISTICS_CUH

#include <cstdint>

class ExecutionStatistic {
public:
    ExecutionStatistic() = default;

    // number of resulted join matches
    uint64_t resultCount{};

    // HJ-specific statistics
    uint64_t buildTime{};
    uint64_t probeTime{};

    // SMJ-specific statistics
    uint64_t sortTime{};
    uint64_t mergeTime{};
};

#endif //STREAMJOINBENCHMARK_EXECUTIONSTATISTICS_CUH
