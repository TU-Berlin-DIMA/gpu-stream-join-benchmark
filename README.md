This is a benchmarking framework for stream join.

## Requirements
The requirements are:

* CMake 3.20 or higher
* A C++14 compatible compiler
* CUDA
* Git

## Dependency
* Install vcpackage in the project root directory (see https://vcpkg.io/en/getting-started.html)
* Install the TBB library from vcpakcage: `./vcpkg/vcpkg install tbb`
* Install the YAML-CPP library from vcpakcage: `./vcpkg/vcpkg install yaml-cpp`

## Building
To build:

```bash
mkdir build && cd build
cmake -DCMAKE_CUDA_COMPILER=<path_to_nvcc> -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake ../
make
```

## Running Experiments
To run an experiment, create a yaml file containing the configuration parameters. Please refer to `config/config-example.yaml` for an example of configuration file as well as possible values for a number of parameters. Once created, start the experiment by running:
```bash
./runExperiment <path_to_yaml_config> <outputFileName>
```

The experiment result is a CSV formatted file, containing the measured metrics for all combination of configuration parameters defined in the yaml file. When enabled in the yaml configuration, the power utilization records is saved as `power_data.txt`, containing a timestamp and power utilization in Watt.