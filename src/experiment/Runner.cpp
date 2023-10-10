#include <experiment/Runner.hpp>
#include <experiment/Config.hpp>
#include <tbb/parallel_invoke.h>
#include <cuda_runtime.h>
#include <sjb/utils/ErrorChecking.cuh>
#include <sjb/source/Source.hpp>
#include <sjb/sink/Sink.h>
#include <sjb/windowing/WindowingFactory.hpp>
#include <sjb/utils/Logger.hpp>
#include <fstream>
#include <sstream>
#include <sjb/utils/PowerMeasurementAgent.hpp>

std::pair<std::stringstream, uint64_t>
Runner::runExperimentWithConfig(ExperimentConfig config, bool trackProgress) {
    // Generate data
    // use pinned memory for the tuple buffers
    Tuple *left, *right;

    // pre create the event streams
    tbb::parallel_invoke(
            [&]() {
                CUDA_CHECK(cudaMallocHost(&left, sizeof(Tuple) * config.dataConfig.dataSize));
                DataGenerator::generate(left, config.dataConfig.dataSize, config.dataConfig.zipfTheta,
                                        config.dataConfig.tsMode, config.dataConfig.numDistinctKey, 123,
                                        config.dataConfig.timestampUpperBound);
            },
            [&]() {
                CUDA_CHECK(cudaMallocHost(&right, sizeof(Tuple) * config.dataConfig.dataSize));
                DataGenerator::generate(right, config.dataConfig.dataSize, config.dataConfig.zipfTheta,
                                        config.dataConfig.tsMode, config.dataConfig.numDistinctKey, 456,
                                        config.dataConfig.timestampUpperBound);
            }
    );

    // create power measurement agent
    auto pma = PowerMeasurementAgent();

    // create a source
    Source source = Source(config.dataConfig.dataSize, config.dataConfig.batchSize);
    // create a sink
    Sink sink;

    sink.counterGPU[0] = 0;
    cudaDeviceSynchronize();

    // create a windowing
    auto windowing = Windowing::WindowingFactory::create(config.algorithmConfig, config.dataConfig, config.queryConfig,
                                                         config.engineConfig, sink);
    if (config.measurePower) {
        pma.start();
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    sink.markProcessingStart();

    source.start(left, right, config.engineConfig.nSourceThreads, windowing);

    auto t1 = std::chrono::high_resolution_clock::now();

    if (config.measurePower) {
        pma.end();
    }

    uint64_t msElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto matchCount = sink.getMatchCount(config.algorithmConfig.resultWritingMethod == Windowing::RESULT_WRITING_METHOD::Atomic);
    auto selectivity = 100.0 * matchCount / sink.getPossibleMatches().load();
    LOG_INFO("Joiner: %s, Match Count: %lu, Selectivity: %.6f%%, ElapsedTime: %lu ms",
             windowing->getAlgorithmName().c_str(), matchCount,
             selectivity,
             msElapsed);

    LOG_INFO("Avg Build: %ld, Avg Probe: %ld", sink.getAverageBuildTime(), sink.getAverageProbeTime());
    LOG_INFO("Avg Sort: %ld, Avg Merge: %ld", sink.getAverageSortTime(), sink.getAverageMergeTime());

    std::stringstream result;

    if (trackProgress) {
        for (const auto progress: sink.getProgress()) {
            result << windowing->getAlgorithmName() << ","
                   << config.dataConfig.batchSize << ","
                   << selectivity << ","
                   << progress.isWindowClosed << ","
                   << progress.currentMatches << "," // n produced tuple
                   << progress.timestamp << "\n"; // ts of production
        }
    } else {
        result << windowing->getAlgorithmName() << ","
               << std::to_string(config.algorithmConfig.resultWritingMethod) << ","
               << config.dataConfig.dataSize << ","
               << config.dataConfig.batchSize << ","
               << config.dataConfig.zipfTheta << ","
               << config.dataConfig.numDistinctKey << ","
               << config.queryConfig.windowSize << ","
               << config.algorithmConfig.cpuhjConfig.nProheThreads << ","
               << config.engineConfig.joinerThreads << ","
               << selectivity << ","
               << msElapsed << "\n";
    }

    cudaFreeHost(left);
    cudaFreeHost(right);

    return std::pair<std::stringstream, uint64_t>(result.str(), msElapsed);
}

void Runner::runExperimentsAndWriteResultToFile(std::vector<ExperimentConfig> configs, std::string outputFileName) {
    std::ofstream experimentOutputFile;
    experimentOutputFile.open(outputFileName, std::ios_base::app);
    if (experimentOutputFile.tellp() == 0) {
        experimentOutputFile
                << "algorithm_name,"
                   "write_result,"
                   "ingestion_rate,"
                   "batch_size,"
                   "zipf_theta,"
                   "distinct_key,"
                   "window_size,"
                   "probe_threads,"
                   "joiner_threads,"
                   "selectivity,"
                   "time\n";
    }

    LOG_INFO("Running a total of %lu configurations", configs.size());
    for (uint64_t i = 0; i < configs.size(); i++) {
        LOG_INFO("Running configuration %lu of %lu..", i + 1, configs.size());

        std::pair<std::stringstream, uint64_t> result;
        uint64_t msElapsed = 0;

        auto currentConfig = configs[i];
        uint64_t delta = std::numeric_limits<int64_t>::max();

        auto latestUnsustainedRate = 0;
        auto nextRate = currentConfig.dataConfig.dataSize;
        auto currentRate = currentConfig.dataConfig.dataSize;
        auto latestSustainedRate = currentConfig.dataConfig.dataSize;
        while (delta > currentConfig.precision) {

            std::vector<uint64_t> msElapsedTimes;

            // Run the repeated the experiment
            for (uint64_t rep = 0; rep < currentConfig.repeat; rep++) {
                result = runExperimentWithConfig(currentConfig);
                msElapsed = result.second;
                msElapsedTimes.push_back(msElapsed);
            }

            // compute the median of elapsed time
            auto median = computeMedian(msElapsedTimes);
            auto isSustained = median < 1000;


            if (isSustained) {
                latestSustainedRate = currentRate;
                if (latestUnsustainedRate == 0) {
                    nextRate *= 2;
                } else {
                    nextRate = (latestUnsustainedRate + currentRate) / 2;
                }
            } else {
                latestUnsustainedRate = currentRate;
                nextRate = (latestSustainedRate + currentRate) / 2;
            }

            // update the delta
            if (currentRate > nextRate) {
                delta = currentRate - nextRate;
            } else {
                delta = nextRate - currentRate;
            }
            LOG_INFO("Latest ms elapsed: %lu dataSize: %lu delta:%lu",
                     msElapsed, currentConfig.dataConfig.dataSize, delta);
            currentRate = nextRate;
            currentConfig.dataConfig.dataSize = currentRate;

            // stop if it is around 1000
            if (abs(median - 1000) <= 50) {
                break;
            }
        }
        if (i < configs.size() - 1) {
            LOG_INFO("Moving on the next configuration");
        }
        LOG_DEBUG("Out: %s", result.first.str().c_str());

        experimentOutputFile << result.first.str();
    }

    experimentOutputFile.close();
}

ExperimentConfig Runner::parseConfigFromFile(std::string) {
    LOG_ERROR("Not Implemented");
    throw std::runtime_error("Not Implemented");
}

double Runner::computeMedian(std::vector<uint64_t> vector) {
    // sort the vector
    std::sort(vector.begin(), vector.end());

    uint64_t median;
    if (vector.size() % 2 == 0) {
        return (vector[vector.size() / 2 - 1] + vector[vector.size() / 2]) / 2;
    } else {
        return vector[vector.size() / 2];
    }
}

void Runner::runProgressivenessExperimentAndWriteResultToFile(std::vector<ExperimentConfig> configs,
                                                              std::string outputFileName) {
    for (uint64_t i = 0; i < configs.size(); i++) {
        LOG_INFO("Running configuration %lu of %lu..", i + 1, configs.size());
        auto result = runExperimentWithConfig(configs[i], true);

        std::ofstream experimentOutputFile;
        experimentOutputFile.open(outputFileName, std::ios_base::app);
        if (experimentOutputFile.tellp() == 0) {
            experimentOutputFile
                    << "algorithm_name,"
                       "batch_size,"
                    "selectivity,"
                    "isWindowClosed,"
                    "match_count,"
                    "timestamp\n";
        }

        experimentOutputFile << result.first.str();
        if (i < configs.size() - 1) {
            LOG_INFO("Moving on the next configuration");
        }
    }
}
