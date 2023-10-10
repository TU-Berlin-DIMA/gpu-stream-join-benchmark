#include <iostream>
#include <cmath>
#include <sjb/utils/Tuple.hpp>
#include <sjb/utils/DataGenerator.hpp>

int main(int argc, char **argv) {
//    uint64_t dataSize = 33554432;
    uint64_t dataSize = 8096;
    Tuple *tuples = static_cast<Tuple *>(std::malloc(dataSize * sizeof(Tuple)));
    float zipfTheta = 0.0;
    uint8_t tsMode = DataGenerator::BURST;

    bool printTsOnly = true;
    DataGenerator::generate(tuples, dataSize, zipfTheta, tsMode, 1024, 12345, 1000);

    for (uint32_t i = 0; i < dataSize; i++) {
        if (printTsOnly) {
            printf("%lu\n", tuples[i].ts);
        } else {
            printf("Key:%u, Val:%u, Ts:%lu\n", tuples[i].key, tuples[i].val, tuples[i].ts);
        }
    }

    return 0;
}