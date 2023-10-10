#include <sjb/utils/Tuple.hpp>
#include <sjb/utils/DataGenerator.hpp>
#include <sjb/source/Source.hpp>
#include <experiment/Config.hpp>
#include <experiment/Runner.hpp>
#include <sstream>
#include <yaml-cpp/yaml.h>
#include <iostream>

#define KiB (1024/sizeof(Tuple))
#define MiB (KiB * 1024)
#define GiB (MiB * 2014)

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "Please run using ./runExperiment <path_to_yaml_config> <outputFileName>." << std::endl;
        return 1;
    }

    std::string yamlConfigPath = argv[1];
    std::string outputFileName = argv[2];

    // DEFAULT PARAMETERS
    ExperimentConfig experimentConfig;
        experimentConfig.dataConfig = {2560 * MiB, 0.0, DataGenerator::TS_MODE::STEADY,
                                   128 * KiB, 1000, 1000};
    experimentConfig.algorithmConfig = {
            Windowing::ALGORITHM_TYPE::HJ,
            Windowing::DEVICE_TYPE::CPU,
            Windowing::PROGRESSIVENESS_TYPE::EAGER,
            Windowing::RESULT_WRITING_METHOD::NoOutput
    };

    experimentConfig.algorithmConfig.cpuhjConfig.nProheThreads = 1;
    experimentConfig.algorithmConfig.cpusmjConfig.nSortThreads = 1;
    experimentConfig.engineConfig = {1024 * MiB, 128, 1, 32, 32 * MiB};
    experimentConfig.queryConfig = {1 * MiB};
    experimentConfig.precision = 32 * KiB;

    // EXPERIMENT PARAMETERS
    // Load the config file
    YAML::Node yamlConfig = YAML::LoadFile(yamlConfigPath);

    // Get the parallelization strategy
    std::vector<Windowing::PARALLELIZATION_STRATEGY> parStrats = {};
    for (const auto &parStrat: yamlConfig["parStrats"].as<std::vector<std::string>>()) {
        if (std::equal(parStrat.begin(), parStrat.end(), "INTRA")) {
            parStrats.push_back(Windowing::PARALLELIZATION_STRATEGY::INTRA);
        } else if (std::equal(parStrat.begin(), parStrat.end(), "INTER")) {
            parStrats.push_back(Windowing::PARALLELIZATION_STRATEGY::INTER);
        } else if (std::equal(parStrat.begin(), parStrat.end(), "HYBRID")) {
            parStrats.push_back(Windowing::PARALLELIZATION_STRATEGY::HYBRID);
        } else {
            throw std::invalid_argument("Unknown parallelization strategy: " + parStrat);
        }
    }

    std::vector<Windowing::ALGORITHM_TYPE> algTypes = {};
    for (const auto &algTypeStr: yamlConfig["algTypes"].as<std::vector<std::string>>()) {
        if (std::equal(algTypeStr.begin(), algTypeStr.end(), "NLJ")) {
            algTypes.push_back(Windowing::ALGORITHM_TYPE::NLJ);
        } else if (std::equal(algTypeStr.begin(), algTypeStr.end(), "HJ")) {
            algTypes.push_back(Windowing::ALGORITHM_TYPE::HJ);
        } else if (std::equal(algTypeStr.begin(), algTypeStr.end(), "SMJ")) {
            algTypes.push_back(Windowing::ALGORITHM_TYPE::SMJ);
        } else {
            throw std::invalid_argument("Unknown algorithm type: " + algTypeStr);
        }
    }


    std::vector<Windowing::DEVICE_TYPE> devTypes = {};
    for (const auto &deviceTypeStr: yamlConfig["deviceTypes"].as<std::vector<std::string>>()) {
        if (std::equal(deviceTypeStr.begin(), deviceTypeStr.end(), "CPU")) {
            devTypes.push_back(Windowing::DEVICE_TYPE::CPU);
        } else if (std::equal(deviceTypeStr.begin(), deviceTypeStr.end(), "GPU")) {
            devTypes.push_back(Windowing::DEVICE_TYPE::GPU);
        } else {
            throw std::invalid_argument("Unknown device type: " + deviceTypeStr);
        }
    }

    std::vector<Windowing::PROGRESSIVENESS_TYPE> progModes = {};
    for (const auto &progressivenessModeStr: yamlConfig["progressivenessModes"].as<std::vector<std::string>>()) {
        if (std::equal(progressivenessModeStr.begin(), progressivenessModeStr.end(), "LAZY")) {
            progModes.push_back(Windowing::PROGRESSIVENESS_TYPE::LAZY);
        } else if (std::equal(progressivenessModeStr.begin(), progressivenessModeStr.end(), "EAGER")) {
            progModes.push_back(Windowing::PROGRESSIVENESS_TYPE::EAGER);
        } else {
            throw std::invalid_argument("Unknown progressiveness mode: " + progressivenessModeStr);
        }
    }

    std::vector<Windowing::RESULT_WRITING_METHOD> resultMaterializationMethods = {};
    for (const auto &resultMaterializationModeStr: yamlConfig["resultMaterializationMethod"].as<std::vector<std::string>>()) {
        if (std::equal(resultMaterializationModeStr.begin(), resultMaterializationModeStr.end(), "NoOutput")) {
            resultMaterializationMethods.push_back(Windowing::RESULT_WRITING_METHOD::NoOutput);
        } else if (std::equal(resultMaterializationModeStr.begin(), resultMaterializationModeStr.end(),
                              "CountKernel")) {
            resultMaterializationMethods.push_back(Windowing::RESULT_WRITING_METHOD::CountKernel);
        } else if (std::equal(resultMaterializationModeStr.begin(), resultMaterializationModeStr.end(),
                              "EstimatedSelectivity")) {
            resultMaterializationMethods.push_back(Windowing::RESULT_WRITING_METHOD::EstimatedSelectivity);
        } else if (std::equal(resultMaterializationModeStr.begin(), resultMaterializationModeStr.end(), "Bitmap")) {
            resultMaterializationMethods.push_back(Windowing::RESULT_WRITING_METHOD::Bitmap);
        } else if (std::equal(resultMaterializationModeStr.begin(), resultMaterializationModeStr.end(), "Atomic")) {
            resultMaterializationMethods.push_back(Windowing::RESULT_WRITING_METHOD::Atomic);
        } else {
            throw std::invalid_argument("Unknown progressiveness mode: " + resultMaterializationModeStr);
        }
    }

    auto numDistinctKeys = yamlConfig["distinctKeys"].as<std::vector<uint64_t>>();

    auto zipfThetas = yamlConfig["zipfThetas"].as<std::vector<float>>();

    auto windowSizes = yamlConfig["windowSizesMB"].as<std::vector<uint64_t>>();
    std::transform(windowSizes.begin(), windowSizes.end(), windowSizes.begin(), [](uint64_t windowSizeMB) {
        return windowSizeMB * MiB;
    });

    auto batchSizes = yamlConfig["batchSizeKB"].as<std::vector<uint64_t>>();
    std::transform(batchSizes.begin(), batchSizes.end(), batchSizes.begin(), [](uint64_t batchSizeKB) {
        return batchSizeKB * KiB;
    });

    auto measurePower = yamlConfig["measurePowerUsage"].as<std::string>();
    if (std::equal(measurePower.begin(), measurePower.end(), "Yes")) {
        experimentConfig.measurePower = true;

    } else if (std::equal(measurePower.begin(), measurePower.end(), "No")) {
        experimentConfig.measurePower = false;
    } else {
        throw std::invalid_argument("Unknown parameter for measure power: " + measurePower);
    }

    experimentConfig.repeat = yamlConfig["repeat"].as<std::uint64_t>();

    Runner runner = Runner();

    std::vector<ExperimentConfig> configurations;

    for (auto algtype: algTypes) {
        for (auto devType: devTypes) {
            for (auto progMode: progModes) {
                for (auto numDistinctKey: numDistinctKeys) {
                    for (auto resultMaterializationMethod: resultMaterializationMethods) {
                        for (auto windowSize: windowSizes) {
                            for (auto batchSize: batchSizes) {
                                for (auto zipfTheta: zipfThetas) {
                                    for (auto parStrat: parStrats) {
                                        experimentConfig.dataConfig.batchSize = batchSize;
                                        experimentConfig.dataConfig.zipfTheta = zipfTheta;
                                        experimentConfig.algorithmConfig.progressiveness = progMode;
                                        experimentConfig.algorithmConfig.device = devType;
                                        experimentConfig.algorithmConfig.underlyingAlgorithm = algtype;
                                        experimentConfig.algorithmConfig.resultWritingMethod = resultMaterializationMethod;
                                        experimentConfig.dataConfig.numDistinctKey = numDistinctKey;
                                        experimentConfig.queryConfig.windowSize = windowSize;
                                        experimentConfig.engineConfig.maxTupleInWindow = windowSize;

                                        // config par strat
                                        if (parStrat == Windowing::PARALLELIZATION_STRATEGY::INTRA) {
                                            experimentConfig.engineConfig.joinerThreads = 1;
                                            experimentConfig.algorithmConfig.cpuhjConfig.nProheThreads = std::thread::hardware_concurrency();
                                            experimentConfig.algorithmConfig.cpusmjConfig.nSortThreads = std::thread::hardware_concurrency();
                                        } else if (parStrat == Windowing::PARALLELIZATION_STRATEGY::INTER) {
                                            experimentConfig.engineConfig.joinerThreads = std::thread::hardware_concurrency();
                                            experimentConfig.algorithmConfig.cpuhjConfig.nProheThreads = 1;
                                            experimentConfig.algorithmConfig.cpusmjConfig.nSortThreads  = 1;
                                        } else if (parStrat == Windowing::PARALLELIZATION_STRATEGY::HYBRID) {
                                            experimentConfig.engineConfig.joinerThreads = 4;
                                            experimentConfig.algorithmConfig.cpuhjConfig.nProheThreads = std::thread::hardware_concurrency()/4;
                                            experimentConfig.algorithmConfig.cpusmjConfig.nSortThreads  = std::thread::hardware_concurrency() / 4;
                                        }

                                        configurations.push_back(experimentConfig);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    auto measuredMetric = yamlConfig["measuredMetric"].as<std::string>();
    if (std::equal(measuredMetric.begin(), measuredMetric.end(), "Throughput")) {
        runner.runExperimentsAndWriteResultToFile(configurations, outputFileName);
    } else if (std::equal(measuredMetric.begin(), measuredMetric.end(), "cumulativePercentage")) {
        runner.runProgressivenessExperimentAndWriteResultToFile(configurations, "progressiveness.csv");
    } else {
        throw std::invalid_argument("Unknown measured metrics: " + measuredMetric);
    }

    return 0;
}
