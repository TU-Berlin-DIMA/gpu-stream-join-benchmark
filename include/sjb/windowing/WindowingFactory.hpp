#ifndef STREAMJOINBENCHMARK_WINDOWINGFACTORY_HPP
#define STREAMJOINBENCHMARK_WINDOWINGFACTORY_HPP

#include <sjb/windowing/BaseWindowing.cuh>
#include <experiment/Config.hpp>

namespace Windowing {
    class WindowingFactory {
        WindowingFactory();

        ~WindowingFactory();

    public:
        static std::shared_ptr<BaseWindowing> create(AlgorithmConfig algorithmConfig,
                                                     DataConfig dataConfig,
                                                     QueryConfig queryConfig,
                                                     EngineConfig engineConfig,
                                                     Sink &sink);

    };

}

#endif //STREAMJOINBENCHMARK_WINDOWINGFACTORY_HPP
