#ifndef STREAMJOINBENCHMARK_POWERMEASUREMENTAGENT_HPP
#define STREAMJOINBENCHMARK_POWERMEASUREMENTAGENT_HPP

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <nvml.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>
#include <string>

#define GPU_TYPE 0

class PowerMeasurementAgent {

public:

    explicit PowerMeasurementAgent();

    static int getNVMLError(nvmlReturn_t resultToCheck);

    static void *powerPollingFunc(void *ptr);

    void start();

    void end();

private:
    std::string outputFilename;
};



#endif //STREAMJOINBENCHMARK_POWERMEASUREMENTAGENT_HPP
