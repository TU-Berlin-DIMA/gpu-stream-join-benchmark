#include <sstream>

#include <sjb/utils/Tuple.hpp>
#include <sjb/utils/DataGenerator.hpp>
#include <sjb/source/Source.hpp>
#include <experiment/Config.hpp>
#include <experiment/Runner.hpp>

#define KiB (1024/sizeof(Tuple))
#define MiB (KiB * 1024)
#define GiB (MiB * 1014)

int main(int argc, char **argv) {
    ExperimentConfig config;

    config.dataConfig = {128 * MiB, 0.99, DataGenerator::TS_MODE::STEADY,
                         16 * KiB, 1000, 1000};
    config.algorithmConfig = {
            Windowing::ALGORITHM_TYPE::SMJ,
            Windowing::DEVICE_TYPE::GPU,
            Windowing::PROGRESSIVENESS_TYPE::LAZY,
            Windowing::RESULT_WRITING_METHOD::NoOutput
    };

    config.algorithmConfig.cpusmjConfig.nSortThreads = 1;
    config.algorithmConfig.cpuhjConfig.nProheThreads = 1;

    config.engineConfig = {256 * MiB,
                           128,
                           1, // for EAGER, the number of join thread should always be one
                           32, // for EAGER, the number of thread should always be one
                           4 * MiB};
    config.queryConfig = {2 * MiB};

    Runner runner = Runner();
    runner.runExperimentWithConfig(config);
//    runner.runProgressivenessExperimentAndWriteResultToFile({config}, "eager_gpu_hj_nm_cumulative_percentage.csv");
    return 0;
}
