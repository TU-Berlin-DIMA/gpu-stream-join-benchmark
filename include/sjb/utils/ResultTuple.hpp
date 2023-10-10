#ifndef STREAMJOINBENCHMARK_RESULTTUPLE_H
#define STREAMJOINBENCHMARK_RESULTTUPLE_H
#include <cstdint>

class ResultTuple {
public:
    uint32_t lkey; // 4bytes
    uint32_t rkey; // 4bytes
    uint32_t lVal; // 4bytes
    uint32_t rVal; // 4bytes
    uint64_t lTs; // 8 bytes
    uint64_t rTs; // 8 bytes
};

#endif //STREAMJOINBENCHMARK_RESULTTUPLE_H
