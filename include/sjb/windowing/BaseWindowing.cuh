#ifndef SJB_CUDA_BASEJOINER_CUH
#define SJB_CUDA_BASEJOINER_CUH

#include <sjb/sink/Sink.h>
#include <sjb/utils/Tuple.hpp>
#include <utility>

namespace Windowing {
    enum ALGORITHM_TYPE : uint8_t {
        EMPTY = 0,
        NLJ = 1,
        HJ = 2,
        SMJ = 3,
        SABER = 4,
        HELLS = 5,
    };

    enum PROGRESSIVENESS_TYPE : uint8_t {
        EAGER = 0,
        LAZY = 1,
    };

    enum RESULT_WRITING_METHOD : uint8_t {
        NoOutput = 0,
        CountKernel = 1,
        EstimatedSelectivity = 2,
        Bitmap = 3,
        Atomic = 4,
    };

    enum DEVICE_TYPE : uint8_t {
        CPU = 0,
        GPU = 1,
    };

    enum PARALLELIZATION_STRATEGY : uint8_t {
        INTER = 0,
        INTRA = 1,
        HYBRID = 2,
    };

    class BaseWindowing {

    public:
        explicit BaseWindowing(Sink &sink, std::string algorithmName) : sink(sink),
                                                                        algorithmName(std::move(algorithmName)) {};

        virtual void onIncomingLeft(Tuple *tupleBuffer) = 0;

        virtual void onIncomingRight(Tuple *tupleBuffer) = 0;

        virtual void onEndOfStream() = 0;

    protected:
        Sink &sink;
        std::string algorithmName;

        bool endOfStream = false;

    public:
        const std::string &getAlgorithmName() const {
            return algorithmName;
        }
    };
}

#endif //SJB_CUDA_BASEJOINER_CUH
