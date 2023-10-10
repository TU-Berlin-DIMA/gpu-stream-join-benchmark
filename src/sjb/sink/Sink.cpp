#include <sjb/sink/Sink.h>
#include <sjb/utils/Logger.hpp>
#include <sjb/utils/ErrorChecking.cuh>
#include <cassert>

Sink::Sink() : counter{0}, possibleMatches(0) {
    LOG_DEBUG("Sink()");
    CUDA_CHECK(cudaMallocHost(&sinkBuffer, sinkBufferSize * sizeof(ResultTuple)));
    CUDA_CHECK(cudaMallocHost(&counterGPU, 1 * sizeof(unsigned long long int)));
    cudaStreamCreate(&resultCopyStream);
}

void Sink::addToCounter(uint64_t currentMatches, bool isWindowClosed) {
    counter.fetch_add(currentMatches);

    auto currentTime = std::chrono::high_resolution_clock::now();
    auto durationInMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(
            (currentTime - processingStartTime)).count();
    progress.emplace_back(
            Progress{counter, isWindowClosed, static_cast<uint64_t>(durationInMicroseconds)});
    LOG_DEBUG("incoming result count: %lu, isWindowClosed:%u, timeFromSinkCreation: %lu, cumulative count:%lu",
             currentMatches, isWindowClosed,durationInMicroseconds, counter.load());
}

uint64_t Sink::getMatchCount(bool useGPUCounter) {
    if (useGPUCounter) {
        return counterGPU[0];
    } else {
        return counter;
    }
}

const std::atomic<uint64_t> &Sink::getPossibleMatches() const {
    return possibleMatches;
}

void Sink::addToPossibleMatches(uint64_t count) {
    possibleMatches.fetch_add(count);
}

void Sink::addToBuildTime(uint64_t time) {
    totalBuildTime.fetch_add(time);
}

void Sink::addToProbeTime(uint64_t time) {
    totalProbeTime.fetch_add(time);
}

void Sink::addToSortTime(uint64_t time) {
    totalSortTime.fetch_add(time);
}

void Sink::addToMergeTime(uint64_t time) {
    totalMergeTime.fetch_add(time);
}

void Sink::addToExecutedWindows(uint64_t numberOfExecutedWindows) {
    totalExecutedWindows.fetch_add(numberOfExecutedWindows);
}

void Sink::addToStatistics(ExecutionStatistic statistic) {

}

const std::atomic<uint64_t> &Sink::getTotalExecutedWindows() const {
    return totalExecutedWindows;
}

uint64_t Sink::getAverageSortTime() {
    return totalSortTime / (totalExecutedWindows);
}

uint64_t Sink::getAverageMergeTime() {
    return totalMergeTime / (totalExecutedWindows);
}

uint64_t Sink::getAverageBuildTime() {
    return totalBuildTime / (totalExecutedWindows);
}

uint64_t Sink::getAverageProbeTime() {
    return totalProbeTime / (totalExecutedWindows);
}

tbb::concurrent_vector<Progress> Sink::getProgress() {
    return progress;
}

void Sink::markProcessingStart() {
    processingStartTime = std::chrono::high_resolution_clock::now();
}

void
Sink::addToStore(uint64_t pos, uint32_t lkey, uint32_t rkey, uint32_t lVal, uint32_t rVal, uint64_t lTs, uint64_t rTs) {
    store[pos % store.size()].lkey = lkey;
    store[pos % store.size()].rkey = rkey;
    store[pos % store.size()].lVal = lVal;
    store[pos % store.size()].rVal = rVal;
    store[pos % store.size()].lTs = lTs;
    store[pos % store.size()].rTs = rTs;
}

void
Sink::incrementCounterAndStore(uint32_t lkey, uint32_t rkey, uint32_t lVal, uint32_t rVal, uint64_t lTs, uint64_t rTs) {
    auto oldCount = counter.fetch_add(1);
    addToStore(oldCount, lkey, rkey, lVal, rVal, lTs, rTs);
}

Sink::~Sink() {
    cudaFreeHost(sinkBuffer);
    cudaStreamDestroy(resultCopyStream);
}

void Sink::incrementCounterAndStore(ResultTuple *resultTuple, uint64_t tupleCount) {
    assert(tupleCount <= sinkBufferSize);
    counter.fetch_add(tupleCount);
    CUDA_CHECK(cudaMemcpyAsync(sinkBuffer, resultTuple, tupleCount * sizeof(ResultTuple), cudaMemcpyDeviceToHost,
                               resultCopyStream));
    cudaStreamSynchronize(resultCopyStream);
}

uint64_t Sink::getSinkBufferSize() const {
    return sinkBufferSize;
}
