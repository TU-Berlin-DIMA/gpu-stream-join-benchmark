#ifndef STREAMJOINBENCHMARK_RUNNER_HPP
#define STREAMJOINBENCHMARK_RUNNER_HPP

#include <string>
#include <vector>

class ExperimentConfig;

class Runner {
public:
    std::pair<std::stringstream, uint64_t> runExperimentWithConfig(ExperimentConfig config, bool trackProgress = false);

    void runExperimentsAndWriteResultToFile(std::vector<ExperimentConfig> configs, std::string outputFileName);

    void runProgressivenessExperimentAndWriteResultToFile(std::vector<ExperimentConfig> configs, std::string outputFileName);
private:
    ExperimentConfig parseConfigFromFile(std::string);

    double computeMedian(std::vector<uint64_t>);
};

#endif //STREAMJOINBENCHMARK_RUNNER_HPP
