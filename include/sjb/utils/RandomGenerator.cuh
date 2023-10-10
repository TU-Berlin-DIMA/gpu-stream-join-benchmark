#ifndef STREAMJOINBENCHMARK_RANDOMGENERATOR_CUH
#define STREAMJOINBENCHMARK_RANDOMGENERATOR_CUH

#include "Tuple.hpp"

class RandomGenerator {
public:
    static void generateZipfian(uint32_t *h_randomNumber, uint64_t count, int upperBound,
                                float theta, uint32_t seed, bool sorted = false);

    static uint64_t *generateSineDistributedData(uint64_t count, uint32_t upperBound);
};


#endif //STREAMJOINBENCHMARK_RANDOMGENERATOR_CUH
