#ifndef SJB_CUDA_DATAGENERATOR_HPP
#define SJB_CUDA_DATAGENERATOR_HPP

#include <sjb/utils/Tuple.hpp>
#include <tbb/concurrent_queue.h>

class DataGenerator {
public:
    static void
    generate(Tuple *tuples, uint64_t count, float zipf_theta, uint8_t tsMode, uint64_t distinctKey, uint32_t seed,
             uint64_t ts_upperbound);

    enum TS_MODE: uint8_t {
        STEADY = 0,
        BURST = 1,
    };
};


#endif //SJB_CUDA_DATAGENERATOR_HPP
