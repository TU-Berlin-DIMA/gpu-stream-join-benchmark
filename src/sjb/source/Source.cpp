#include <sjb/source/Source.hpp>
#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>
#include <cassert>
#include <cmath>
#include <sjb/utils/Logger.hpp>
#include <tbb/parallel_invoke.h>

Source::Source(uint64_t count, uint64_t bufferSize) : count(count), bufferSize(bufferSize) {
}

void Source::start(Tuple *leftTuples,
                   Tuple *rightTuples,
                   uint64_t sourceThreads,
                   const std::shared_ptr<Windowing::BaseWindowing> &windowing) {

    tbb::task_arena ingestion(sourceThreads,1, tbb::task_arena::priority::low);
    std::atomic<uint64_t> tupleBufferIndex = {0};
    ingestion.execute([&]() {
                          tbb::parallel_for(
                                  tbb::blocked_range<size_t>(0, count / bufferSize),
                                  [&](const tbb::blocked_range<size_t> &r) {
                                      for (size_t i = r.begin(); i < r.end(); ++i) {
                                          auto oldCount = tupleBufferIndex.fetch_add(1);
                                          windowing->onIncomingLeft(leftTuples + oldCount * bufferSize);
                                          windowing->onIncomingRight(rightTuples + oldCount * bufferSize);
                                      }
                                  }
                          );
                      }
    );

    windowing->onEndOfStream();
}
