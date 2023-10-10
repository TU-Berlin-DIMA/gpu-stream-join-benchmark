#include <sjb/utils/PowerMeasurementAgent.hpp>
#include <chrono>
#include <sjb/utils/Logger.hpp>
#include <utility>

/*
These may be encompassed in a class if desired. Trivial CUDA programs written for the purpose of benchmarking might prefer this approach.
*/
bool pollThreadStatus = false;
unsigned int deviceCount = 0;
char deviceNameStr[64];

nvmlReturn_t nvmlResult;
nvmlDevice_t nvmlDeviceID;
nvmlPciInfo_t nvmPCIInfo;
nvmlEnableState_t pmmode = NVML_FEATURE_ENABLED;
nvmlComputeMode_t computeMode;

pthread_t powerPollThread;

float getPowerConsumption() {
    const char *cmd = "cat /sys/bus/i2c/drivers/ina3221x/1-0040/iio\:device0/in_power0_input";

    const int buf_size = 1024;
    char buf[buf_size];
    float power = 0.0f;

    FILE *pipe = popen(cmd, "r");
    if (!pipe) {
        LOG_ERROR("Error: Failed to execute command: %s", cmd);
        return -1.0f;
    }

    while (fgets(buf, buf_size, pipe)) {
        char *power_str = strstr(buf, "");
        if (power_str) {
            sscanf(power_str, "%f", &power);
            break;
        }
    }

    pclose(pipe);

    return power;
}

/*
Poll the GPU using nvml APIs.
*/
void *PowerMeasurementAgent::powerPollingFunc(void *ptr) {
    unsigned int nvmlPowerReading = 0;
    float powerInFloat;
    FILE *fp = fopen("power_data.txt", "a+");

    while (pollThreadStatus) {
        pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, 0);
        if (GPU_TYPE == 0 && pmmode == NVML_FEATURE_ENABLED) {

            // Get the power management mode of the GPU.
            nvmlResult = nvmlDeviceGetPowerManagementMode(nvmlDeviceID, &pmmode);

            // The following function may be utilized to handle errors as needed.
            PowerMeasurementAgent::getNVMLError(nvmlResult);

            // Check if power management mode is enabled.
            // Get the power usage in milliWatts.
            nvmlResult = nvmlDeviceGetPowerUsage(nvmlDeviceID, &nvmlPowerReading);
            powerInFloat = static_cast<float>(nvmlPowerReading);
        } else if (GPU_TYPE == 1) {
            powerInFloat = getPowerConsumption(); // readings is in miliWatts
        } else {
            LOG_ERROR("Unknown GPU type or NVML Feature is not enabled.");
        }

        auto currentTs = std::chrono::high_resolution_clock::now();

        // The output file stores power in Watts.
        fprintf(fp, "%ld,%.3lf\n",
                std::chrono::duration_cast<std::chrono::milliseconds>(currentTs.time_since_epoch()).count(),
                (powerInFloat) / 1000.0);
        pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
    }

    fclose(fp);
    pthread_exit(0);
}

PowerMeasurementAgent::PowerMeasurementAgent() = default;

/*
Start power measurement by spawning a pthread that polls the GPU.
Function needs to be modified as per usage to handle errors as seen fit.
*/
void PowerMeasurementAgent::start() {
    if (GPU_TYPE == 0) {
        int i;

        // Initialize nvml.
        nvmlResult = nvmlInit();
        if (NVML_SUCCESS != nvmlResult) {
            printf("NVML Init fail: %s\n", nvmlErrorString(nvmlResult));
            exit(0);
        }

        // Count the number of GPUs available.
        nvmlResult = nvmlDeviceGetCount(&deviceCount);
        if (NVML_SUCCESS != nvmlResult) {
            printf("Failed to query device count: %s\n", nvmlErrorString(nvmlResult));
            exit(0);
        }

        for (i = 0; i < deviceCount; i++) {
            // Get the device ID.
            nvmlResult = nvmlDeviceGetHandleByIndex(i, &nvmlDeviceID);
            if (NVML_SUCCESS != nvmlResult) {
                printf("Failed to get handle for device %d: %s\n", i, nvmlErrorString(nvmlResult));
                exit(0);
            }

            // Get the name of the device.
            nvmlResult = nvmlDeviceGetName(nvmlDeviceID, deviceNameStr,
                                           sizeof(deviceNameStr) / sizeof(deviceNameStr[0]));
            if (NVML_SUCCESS != nvmlResult) {
                printf("Failed to get name of device %d: %s\n", i, nvmlErrorString(nvmlResult));
                exit(0);
            }

            // Get PCI information of the device.
            nvmlResult = nvmlDeviceGetPciInfo(nvmlDeviceID, &nvmPCIInfo);
            if (NVML_SUCCESS != nvmlResult) {
                printf("Failed to get PCI info of device %d: %s\n", i, nvmlErrorString(nvmlResult));
                exit(0);
            }

            // Get the compute mode of the device which indicates CUDA capabilities.
            nvmlResult = nvmlDeviceGetComputeMode(nvmlDeviceID, &computeMode);
            if (NVML_ERROR_NOT_SUPPORTED == nvmlResult) {
                printf("This is not a CUDA-capable device.\n");
            } else if (NVML_SUCCESS != nvmlResult) {
                printf("Failed to get compute mode for device %i: %s\n", i, nvmlErrorString(nvmlResult));
                exit(0);
            }
        }

        // This statement assumes that the first indexed GPU will be used.
        // If there are multiple GPUs that can be used by the system, this needs to be done with care.
        // Test thoroughly and ensure the correct device ID is being used.
        nvmlResult = nvmlDeviceGetHandleByIndex(0, &nvmlDeviceID);
    }

    pollThreadStatus = true;

    std::string message = "PowerMeasurement";
    int iret = pthread_create(&powerPollThread, NULL, powerPollingFunc, &message);
    if (iret) {
        fprintf(stderr, "Error - pthread_create() return code: %d\n", iret);
        exit(0);
    }
}

/*
End power measurement. This ends the polling thread.
*/
void PowerMeasurementAgent::end() {
    pollThreadStatus = false;
    pthread_join(powerPollThread, NULL);

    if (GPU_TYPE == 0) {
        nvmlResult = nvmlShutdown();
        if (NVML_SUCCESS != nvmlResult) {
            printf("Failed to shut down NVML: %s\n", nvmlErrorString(nvmlResult));
            exit(0);
        }
    }
}

/*
Return a number with a specific meaning. This number needs to be interpreted and handled appropriately.
*/
int PowerMeasurementAgent::getNVMLError(nvmlReturn_t resultToCheck) {
    if (resultToCheck == NVML_ERROR_UNINITIALIZED)
        return 1;
    if (resultToCheck == NVML_ERROR_INVALID_ARGUMENT)
        return 2;
    if (resultToCheck == NVML_ERROR_NOT_SUPPORTED)
        return 3;
    if (resultToCheck == NVML_ERROR_NO_PERMISSION)
        return 4;
    if (resultToCheck == NVML_ERROR_ALREADY_INITIALIZED)
        return 5;
    if (resultToCheck == NVML_ERROR_NOT_FOUND)
        return 6;
    if (resultToCheck == NVML_ERROR_INSUFFICIENT_SIZE)
        return 7;
    if (resultToCheck == NVML_ERROR_INSUFFICIENT_POWER)
        return 8;
    if (resultToCheck == NVML_ERROR_DRIVER_NOT_LOADED)
        return 9;
    if (resultToCheck == NVML_ERROR_TIMEOUT)
        return 10;
    if (resultToCheck == NVML_ERROR_IRQ_ISSUE)
        return 11;
    if (resultToCheck == NVML_ERROR_LIBRARY_NOT_FOUND)
        return 12;
    if (resultToCheck == NVML_ERROR_FUNCTION_NOT_FOUND)
        return 13;
    if (resultToCheck == NVML_ERROR_CORRUPTED_INFOROM)
        return 14;
    if (resultToCheck == NVML_ERROR_GPU_IS_LOST)
        return 15;
    if (resultToCheck == NVML_ERROR_UNKNOWN)
        return 16;

    return 0;
}

