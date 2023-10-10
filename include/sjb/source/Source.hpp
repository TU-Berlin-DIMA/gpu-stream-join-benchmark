#ifndef STREAMJOINBENCHMARK_SOURCE_HPP
#define STREAMJOINBENCHMARK_SOURCE_HPP

#include <sjb/utils/Tuple.hpp>
#include <tbb/concurrent_queue.h>
#include <sjb/windowing/BaseWindowing.cuh>
#include <tbb/concurrent_queue.h>

class Source {
public:
    enum SIDE {
        LEFT = 0,
        RIGHT = 1
    };

    explicit Source(uint64_t count, uint64_t bufferSize);

    void start(Tuple *leftTuples,
               Tuple *rightTuples,
               uint64_t sourceThreads,
               const std::shared_ptr<Windowing::BaseWindowing> &windowing);

private:
    Tuple **leftBatches;
    Tuple **rightBatches;
    uint64_t count;
    uint64_t bufferSize;
};

#endif //STREAMJOINBENCHMARK_SOURCE_HPP
