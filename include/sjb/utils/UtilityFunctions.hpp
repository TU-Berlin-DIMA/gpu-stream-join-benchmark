#ifndef STREAMJOINBENCHMARK_UTILITYFUNCTIONS_HPP
#define STREAMJOINBENCHMARK_UTILITYFUNCTIONS_HPP

#include <tbb/concurrent_queue.h>
#include <sjb/utils/Tuple.hpp>

class UtilityFunctions {
public:
    static void printTuplesToFile(Tuple *tuples, uint64_t size);
};

#endif //STREAMJOINBENCHMARK_UTILITYFUNCTIONS_HPP
