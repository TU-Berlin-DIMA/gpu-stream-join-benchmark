#ifndef SJB_CUDA_TUPLE_H
#define SJB_CUDA_TUPLE_H

#include <cstdint>

class Tuple {
public:
    uint32_t key;
    uint32_t val;
    uint64_t ts;
};


#endif //SJB_CUDA_TUPLE_H
