#include <sjb/windowing/WindowingFactory.hpp>
#include <sjb/windowing/lazy/executor/CPU/CPU_NLJ.hpp>
#include <sjb/windowing/lazy/executor/CPU/CPU_SMJ.hpp>
#include <sjb/windowing/lazy/executor/CPU/CPU_HJ.hpp>
#include <sjb/windowing/lazy/executor/GPU/GPU_NLJ_NoOutput.cuh>
#include <sjb/windowing/lazy/executor/GPU/GPU_NLJ_CountKernel.cuh>
#include <sjb/windowing/lazy/executor/GPU/GPU_NLJ_EstimatedSelectivity.cuh>
#include <sjb/windowing/lazy/executor/GPU/GPU_NLJ_Bitmap.cuh>
#include <sjb/windowing/lazy/executor/GPU/GPU_HJ_NoOutput.cuh>
#include <sjb/windowing/lazy/executor/GPU/GPU_HJ_CountKernel.cuh>
#include <sjb/windowing/lazy/executor/GPU/GPU_HJ_EstimatedSelectivity.cuh>
#include <sjb/windowing/lazy/executor/GPU/GPU_SMJ_NoOutput.cuh>
#include <sjb/windowing/lazy/executor/GPU/GPU_SMJ_CountKernel.cuh>
#include <sjb/windowing/lazy/executor/GPU/GPU_SMJ_EstimatedSelectivity.cuh>
//#include <windowing/lazy/executor/GPU/GPU_SMJ_EstimatedSelectivity.cuh>
//#include <windowing/lazy/executor/GPU/GPU_SABER.cuh>
//#include <sjb/windowing/lazy/executor/GPU/GPU_HELLS.cuh>
#include <sjb/windowing/lazy/CountBased.cuh>
#include <sjb/windowing/lazy/executor/GPU/GPU_SMJ_EstimatedSelectivity.cuh>
#include <sjb/windowing/eager//CountBased.cuh>
#include <sjb/windowing/eager/executor/CPU/CPU_HJ.hpp>
#include <sjb/windowing/eager/executor/GPU/GPU_HJ.cuh>
//#include <sjb/windowing/eager/executor/GPU/GPU_PK_Empty.cuh>
//#include <sjb/windowing/eager/executor/GPU/GPU_HJ_PK.cuh>
#include <sjb/windowing/lazy/executor/GPU/GPU_HJ_Atomic.cuh>
#include <sjb/windowing/lazy/executor/GPU/GPU_SMJ_Atomic.cuh>
#include <sjb/windowing/eager/executor/GPU/GPU_NLJ.cuh>
#include <sjb/windowing/eager/executor/GPU/GPU_HJ_Atomic.cuh>
#include <sjb/windowing/eager/executor/GPU/GPU_NLJ_BM.cuh>


namespace Windowing {
    WindowingFactory::WindowingFactory() {}

    WindowingFactory::~WindowingFactory() {}

    /**
     * @brief Estimating selectivity using information about number of distinct keys & window size
         * In real world application, the information about the number of distinct keys may
         * not always available
     * @param numDistinctKey the number of distinct key
     * @return the estimated selectivity
     */
    double getEstimatedSelectivity(uint64_t numDistinctKey) {
        /**

         */
        float estimatedSelectivity = 1.0 / numDistinctKey;
        return estimatedSelectivity;
    }

    std::shared_ptr<BaseWindowing>
    WindowingFactory::create(AlgorithmConfig algorithmConfig,
                             DataConfig dataConfig,
                             QueryConfig queryConfig,
                             EngineConfig engineConfig,
                             Sink &sink) {

        std::string algorithmName = "NOTSET";

        if (algorithmConfig.progressiveness == Windowing::PROGRESSIVENESS_TYPE::LAZY) {
            tbb::concurrent_queue<std::shared_ptr<Lazy::Executor::BaseLazyExecutor>> joiners;
            if (algorithmConfig.device == Windowing::DEVICE_TYPE::CPU) {
                // generate CPUs
                if (algorithmConfig.underlyingAlgorithm == Windowing::ALGORITHM_TYPE::NLJ) {
                    algorithmName = "lazy_cpu_nlj_nm";
                    for (uint32_t i = 0; i < engineConfig.joinerThreads; i++) {
                        joiners.push(std::make_shared<Lazy::Executor::CPU_NLJ>(dataConfig.batchSize));
                    }
                } else if (algorithmConfig.underlyingAlgorithm == Windowing::ALGORITHM_TYPE::HJ) {
                    algorithmName = "lazy_cpu_hj_nm";
                    for (uint32_t i = 0; i < engineConfig.joinerThreads; i++) {
                        joiners.push(std::make_shared<Lazy::Executor::CPU_HJ>(dataConfig.numDistinctKey,
                                                                              dataConfig.batchSize,
                                                                              algorithmConfig.cpuhjConfig.nProheThreads,
                                                                              engineConfig.maxTupleInWindow));
                    }
                } else if (algorithmConfig.underlyingAlgorithm == Windowing::ALGORITHM_TYPE::SMJ) {
                    algorithmName = "lazy_cpu_smj";
                    for (uint32_t i = 0; i < engineConfig.joinerThreads; i++) {
                        joiners.push(std::make_shared<Lazy::Executor::CPU_SMJ>(dataConfig.numDistinctKey,
                                                                               dataConfig.batchSize,
                                                                               algorithmConfig.cpusmjConfig.nSortThreads));
                    }
                } else {
                    throw std::runtime_error("Not Implemented");
                }
            } else if (algorithmConfig.device == Windowing::DEVICE_TYPE::GPU) {
                // generate GPUs
                if (algorithmConfig.underlyingAlgorithm == Windowing::ALGORITHM_TYPE::NLJ) {
                    switch (algorithmConfig.resultWritingMethod) {
                        case Windowing::RESULT_WRITING_METHOD::NoOutput:
                            algorithmName = "lazy_gpu_nlj_nm";
                            for (uint32_t i = 0; i < engineConfig.joinerThreads; i++) {
                                joiners.push(std::make_shared<Lazy::Executor::GPU_NLJ_NoOutput>(dataConfig.batchSize,
                                                                                                engineConfig.maxTupleInWindow));
                            }
                            break;
                        case Windowing::RESULT_WRITING_METHOD::CountKernel:
                            algorithmName = "lazy_gpu_nlj_ck";
                            for (uint32_t i = 0; i < engineConfig.joinerThreads; i++) {
                                joiners.push(std::make_shared<Lazy::Executor::GPU_NLJ_CountKernel>(dataConfig.batchSize,
                                                                                                   engineConfig.maxTupleInWindow));
                            }
                            break;
                        case Windowing::RESULT_WRITING_METHOD::EstimatedSelectivity:
                            algorithmName = "lazy_gpu_nlj_es";
                            for (uint32_t i = 0; i < engineConfig.joinerThreads; i++) {
                                joiners.push(std::make_shared<Lazy::Executor::GPU_NLJ_EstimatedSelectivity>(
                                        dataConfig.batchSize,
                                        engineConfig.maxTupleInWindow,
                                        getEstimatedSelectivity(dataConfig.numDistinctKey)));
                            }
                            break;
                        case Windowing::RESULT_WRITING_METHOD::Bitmap:
                            algorithmName = "lazy_gpu_nlj_bm";
                            for (uint32_t i = 0; i < engineConfig.joinerThreads; i++) {
                                joiners.push(std::make_shared<Lazy::Executor::GPU_NLJ_Bitmap>(dataConfig.batchSize,
                                                                                              engineConfig.maxTupleInWindow));
                            }
                            break;
                        default:
                            throw std::runtime_error("Not implemented");
                    }
                } else if (algorithmConfig.underlyingAlgorithm == Windowing::ALGORITHM_TYPE::HJ) {
                    if (algorithmConfig.resultWritingMethod == Windowing::RESULT_WRITING_METHOD::NoOutput) {
                        algorithmName = "lazy_gpu_hj_nm";
                        for (uint32_t i = 0; i < engineConfig.joinerThreads; i++) {
                            joiners.push(std::make_shared<Lazy::Executor::GPU_HJ_NoOutput>(dataConfig.numDistinctKey,
                                                                                           dataConfig.batchSize,
                                                                                           engineConfig.maxTupleInWindow));
                        }
                    } else if (algorithmConfig.resultWritingMethod == Windowing::RESULT_WRITING_METHOD::CountKernel) {
                        algorithmName = "lazy_gpu_hj_ck";
                        for (uint32_t i = 0; i < engineConfig.joinerThreads; i++) {
                            joiners.push(std::make_shared<Lazy::Executor::GPU_HJ_CountKernel>(dataConfig.numDistinctKey,
                                                                                              dataConfig.batchSize,
                                                                                              engineConfig.maxTupleInWindow));
                        }
                    } else if (algorithmConfig.resultWritingMethod ==
                               Windowing::RESULT_WRITING_METHOD::EstimatedSelectivity) {
                        algorithmName = "lazy_gpu_hj_es";
                        for (uint32_t i = 0; i < engineConfig.joinerThreads; i++) {
                            joiners.push(std::make_shared<Lazy::Executor::GPU_HJ_EstimatedSelectivity>(
                                    getEstimatedSelectivity(dataConfig.numDistinctKey),
                                    dataConfig.numDistinctKey,
                                    dataConfig.batchSize,
                                    engineConfig.maxTupleInWindow));
                        }
                    } else if (algorithmConfig.resultWritingMethod ==
                             Windowing::RESULT_WRITING_METHOD::Atomic) {
                        algorithmName = "lazy_gpu_hj_at";
                        for (uint32_t i = 0; i < engineConfig.joinerThreads; i++) {
                            joiners.push(std::make_shared<Lazy::Executor::GPU_HJ_Atomic>(
                                    dataConfig.numDistinctKey,
                                    dataConfig.batchSize,
                                    engineConfig.maxTupleInWindow));
                        }
                    }
                } else if (algorithmConfig.underlyingAlgorithm == Windowing::ALGORITHM_TYPE::SMJ) {

                    if (algorithmConfig.resultWritingMethod == Windowing::RESULT_WRITING_METHOD::NoOutput) {
                        algorithmName = "lazy_gpu_smj_nm";
                        for (uint32_t i = 0; i < engineConfig.joinerThreads; i++) {
                            joiners.push(std::make_shared<Lazy::Executor::GPU_SMJ_NoOutput>(
                                    dataConfig.numDistinctKey,
                                    dataConfig.batchSize,
                                    engineConfig.maxTupleInWindow));
                        }
                    } else if (algorithmConfig.resultWritingMethod == Windowing::RESULT_WRITING_METHOD::CountKernel) {
                        algorithmName = "lazy_gpu_smj_ck";
                        for (uint32_t i = 0; i < engineConfig.joinerThreads; i++) {
                            joiners.push(std::make_shared<Lazy::Executor::GPU_SMJ_CountKernel>(
                                    dataConfig.numDistinctKey,
                                    dataConfig.batchSize,
                                    engineConfig.maxTupleInWindow));
                        }
                    } else if (algorithmConfig.resultWritingMethod ==
                               Windowing::RESULT_WRITING_METHOD::EstimatedSelectivity) {
                        auto estimatedSelectivity = getEstimatedSelectivity(dataConfig.numDistinctKey);

                        algorithmName = "lazy_gpu_smj_es";
                        for (uint32_t i = 0; i < engineConfig.joinerThreads; i++) {
                            joiners.push(std::make_shared<Lazy::Executor::GPU_SMJ_EstimatedSelectivity>(
                                    dataConfig.numDistinctKey,
                                    dataConfig.batchSize,
                                    engineConfig.maxTupleInWindow, estimatedSelectivity));
                        }
                    } else if (algorithmConfig.resultWritingMethod == Windowing::RESULT_WRITING_METHOD::Atomic) {
                        algorithmName = "lazy_gpu_smj_at";
                        for (uint32_t i = 0; i < engineConfig.joinerThreads; i++) {
                            joiners.push(std::make_shared<Lazy::Executor::GPU_SMJ_Atomic>(
                                    dataConfig.numDistinctKey,
                                    dataConfig.batchSize,
                                    engineConfig.maxTupleInWindow));
                        }
                    }


                } else if (algorithmConfig.underlyingAlgorithm == Windowing::ALGORITHM_TYPE::SABER) {
                    for (uint32_t i = 0; i < engineConfig.joinerThreads; i++) {
//                        joiners.push(std::make_shared<Lazy::Executor::GPU_SABER>(dataConfig.batchSize));
                    }
                } else if (algorithmConfig.underlyingAlgorithm == Windowing::ALGORITHM_TYPE::HELLS) {
                    algorithmName = "lazy_gpu_nlj_bitmap";
//                    for (uint32_t i = 0; i < engineConfig.joinerThreads; i++) {
//                        joiners.push(std::make_shared<Lazy::Executor::GPU_HELLS>(algorithmConfig.writeResultToSink,
//                                                                                 dataConfig.batchSize));
//                    }
                } else {
                    throw std::runtime_error("Not Implemented");
                }
            } else {
                throw std::runtime_error("Not Implemented");
            }
            return std::make_shared<Windowing::Lazy::CountBased>(sink, engineConfig.ringBufferSize,
                                                                 engineConfig.ringBufferMaxTries,
                                                                 queryConfig.windowSize, dataConfig.batchSize,
                                                                 joiners, algorithmName);
        } else {
            std::shared_ptr<Eager::Executor::BaseEagerExecutor> joiner;

            if (algorithmConfig.device == Windowing::DEVICE_TYPE::CPU) {
                if (algorithmConfig.underlyingAlgorithm == Windowing::ALGORITHM_TYPE::HJ) {
                    algorithmName = "eager_cpu_hj_nm";
                    joiner = std::make_shared<Eager::Executor::CPU_HJ>(engineConfig.maxTupleInWindow,
                                                                       dataConfig.numDistinctKey,
                                                                       dataConfig.batchSize);
                }
            } else if (algorithmConfig.device == Windowing::DEVICE_TYPE::GPU) {
                if (algorithmConfig.underlyingAlgorithm == Windowing::ALGORITHM_TYPE::EMPTY) {
                    if (algorithmConfig.eagerJoinConfig.usePK) {
//                        algorithmName = "eager_gpu_empty_nm_pk";
//                        joiner = std::make_shared<Eager::Executor::GPU_PK_Empty>(dataConfig.numDistinctKey,
//                                                                              queryConfig.windowSize,
//                                                                              dataConfig.batchSize);
                        LOG_ERROR("Not Implemented: GPU EMPTY without PK");
                        throw std::runtime_error("Not Implemented");
                    } else {
                        LOG_ERROR("Not Implemented: GPU EMPTY without PK");
                        throw std::runtime_error("Not Implemented");
                    }
                } else if (algorithmConfig.underlyingAlgorithm == Windowing::ALGORITHM_TYPE::HJ) {

                    if (algorithmConfig.eagerJoinConfig.usePK) {
//                        algorithmName = "eager_gpu_hj_nm_pk";
//                        joiner = std::make_shared<Eager::Executor::GPU_HJ_PK>(dataConfig.numDistinctKey,
//                                                                              queryConfig.windowSize,
//                                                                              dataConfig.batchSize);
                        LOG_ERROR("Not Implemented: GPU EMPTY without PK");
                        throw std::runtime_error("Not Implemented");
                    } else {

                        if (algorithmConfig.resultWritingMethod == Windowing::RESULT_WRITING_METHOD::NoOutput) {
                            algorithmName = "eager_gpu_hj_nm";
                            joiner = std::make_shared<Eager::Executor::GPU_HJ>(dataConfig.numDistinctKey,
                                                                               queryConfig.windowSize,dataConfig.batchSize);
                        } else if (algorithmConfig.resultWritingMethod == Windowing::RESULT_WRITING_METHOD::Atomic) {
                            algorithmName = "eager_gpu_hj_at";
                            joiner = std::make_shared<Eager::Executor::GPU_HJ_Atomic>(dataConfig.numDistinctKey,
                                                                               queryConfig.windowSize,dataConfig.batchSize);
                        } else {
                            LOG_ERROR("Not Implemented");
                            throw std::runtime_error("Not Implemented");
                        }


                    }
                }else if (algorithmConfig.underlyingAlgorithm == Windowing::ALGORITHM_TYPE::NLJ) {

                    if (algorithmConfig.eagerJoinConfig.usePK) {
                        LOG_ERROR("Not Implemented: GPU NLJ with PK");
                        throw std::runtime_error("Not Implemented");
                    } else {
                        if (algorithmConfig.resultWritingMethod == Windowing::RESULT_WRITING_METHOD::NoOutput) {
                            algorithmName = "eager_gpu_nlj_nm";
                            joiner = std::make_shared<Eager::Executor::GPU_NLJ>(dataConfig.batchSize, queryConfig.windowSize);
                        } if (algorithmConfig.resultWritingMethod == Windowing::RESULT_WRITING_METHOD::Bitmap) {
                            algorithmName = "eager_gpu_nlj_bm";
                            joiner = std::make_shared<Eager::Executor::GPU_NLJ_BM>(dataConfig.batchSize, queryConfig.windowSize);
                        } else {
                            LOG_ERROR("Not Implemented");
                            throw std::runtime_error("Not Implemented");
                        }

                    }
                }
            } else {
                throw std::runtime_error("Not Implemented");
            }

            return std::make_shared<Windowing::Eager::CountBased>(sink,
                                                                  queryConfig.windowSize,
                                                                  dataConfig.batchSize,
                                                                  joiner,
                                                                  algorithmName);
        }
    }
} // namespace Windowing