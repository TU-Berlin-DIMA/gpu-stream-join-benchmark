#include <algorithm>
#include <sjb/utils/DataGenerator.hpp>
#include <random>
#include <sjb/utils/RandomGenerator.cuh>
#include <sjb/utils/Logger.hpp>
#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <sjb/utils/ErrorChecking.cuh>

void DataGenerator::generate(Tuple *tuples, uint64_t count, float zipf_theta, uint8_t tsMode, uint64_t distinctKey,
                             uint32_t seed, uint64_t ts_upperbound) {
    tbb::task_arena datagen;
    // generate timestamp
    std::vector<uint64_t> timestamps(count);
    if (tsMode == DataGenerator::BURST) {
        LOG_DEBUG("Generating data with BURST timestamp.");

        uint64_t *sineDistributedTs = RandomGenerator::generateSineDistributedData(count, ts_upperbound);
        datagen.execute([&timestamps, &count, &sineDistributedTs]() {
            tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, count),
                    [&timestamps, &sineDistributedTs](const tbb::blocked_range<size_t> &r) {
                        for (size_t i = r.begin(); i < r.end(); ++i) {
                            timestamps[i] = sineDistributedTs[i];
                        }
                    }
            );
        });
    } else if (tsMode == STEADY) {
        LOG_DEBUG("Generating data with STEADY timestamp.");

        assert(count > ts_upperbound);
        auto tupleCountPerMs = std::floor(count / ts_upperbound);

        datagen.execute([&timestamps, &count, &tupleCountPerMs]() {
            tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, count),
                    [&timestamps, &tupleCountPerMs](const tbb::blocked_range<size_t> &r) {
                        for (size_t i = r.begin(); i < r.end(); ++i) {
                            timestamps[i] = std::floor(1.0 * i / (tupleCountPerMs));
                        }
                    }
            );
        });
    } else {
        LOG_ERROR("Unknown TS_MODE");
        exit(1);
    }

    // generate keys and values
    uint32_t *h_randomNumbers;
    CUDA_CHECK(cudaMallocHost(&h_randomNumbers, count * sizeof(uint32_t)));
    RandomGenerator::generateZipfian(h_randomNumbers, count, distinctKey, zipf_theta, seed);

    // Copy the random numbers to tuple buffers
    std::vector<uint64_t> tupleIndex(count);
    std::iota(tupleIndex.begin(), tupleIndex.end(), 0);

    datagen.execute([&tuples, &tupleIndex, &h_randomNumbers, &timestamps]() {
        tbb::parallel_for_each(tupleIndex.begin(), tupleIndex.end(),
                               [&tuples, &h_randomNumbers, &timestamps](const uint64_t i) {
                                   tuples[i].key = h_randomNumbers[i];
                                   tuples[i].val = h_randomNumbers[i];
                                   tuples[i].ts = timestamps[i];
                               });
    });

    // cleanup
    CUDA_CHECK(cudaFreeHost(h_randomNumbers));
}
