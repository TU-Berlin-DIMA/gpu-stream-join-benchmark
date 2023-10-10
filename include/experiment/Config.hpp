#ifndef STREAMJOINBENCHMARK_CONFIG_HPP
#define STREAMJOINBENCHMARK_CONFIG_HPP

#include <sjb/utils/DataGenerator.hpp>

struct CPUHJConfig {
    uint64_t nProheThreads = 1;
};

struct CPUSMJConfig {
    uint64_t nSortThreads = 1;
};

struct EagerJoinConfig {
    bool usePK = false;
};


struct DataConfig {
    uint64_t dataSize;
    float zipfTheta;
    uint8_t  tsMode;
    uint64_t batchSize;
    uint64_t numDistinctKey;

    uint64_t timestampUpperBound = 1000; // i.e., each timestamp is a millisecond, and all data are within 1 second
};

struct QueryConfig {
    uint64_t windowSize;
};

struct AlgorithmConfig {
    uint8_t underlyingAlgorithm;
    uint8_t device;
    uint8_t progressiveness;

    uint8_t resultWritingMethod = 0;

    CPUHJConfig cpuhjConfig;
    CPUSMJConfig cpusmjConfig;

    EagerJoinConfig eagerJoinConfig;
};

struct EngineConfig {
    uint64_t ringBufferSize;
    uint64_t ringBufferMaxTries;

    uint64_t joinerThreads;
    uint64_t nSourceThreads;

    uint64_t maxTupleInWindow;
};

struct ExperimentConfig {
    DataConfig dataConfig;
    QueryConfig queryConfig;
    AlgorithmConfig algorithmConfig;
    EngineConfig engineConfig;

    uint64_t precision = 4096;
    uint64_t repeat = 1;
    bool measurePower = false;
};

#endif //STREAMJOINBENCHMARK_CONFIG_HPP
