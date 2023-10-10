#include <sjb/utils/RandomGenerator.cuh>
#include <sjb/utils/ErrorChecking.cuh>
#include <stdio.h>

#include <curand.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

// https://github.com/llersch/cpp_random_distributions/blob/master/zipfian_int_distribution.h
__device__ float d_zeta(uint64_t upperBound, float theta) {
    float ans = 0.0;
    for (unsigned long i = 1; i <= upperBound; ++i) {
        ans += std::pow(1.0 / i, theta);
    }
    return ans;
}

__global__ void zetaKernel(uint64_t upperBound, float theta, float *ans) {
    auto i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < upperBound) {
        atomicAdd(ans, std::pow(1.0 / i, theta));
    }
}

// "Quickly Generating Billion-Record Synthetic Databases", Gray et al,
__global__ void zipfKernel(uint32_t *d_randomNumber, const uint64_t count, const uint32_t upperBound,
                           const float theta, float zetan, const float_t *randomFloat) {
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < count) {
        float alpha = 1 / (1 - theta);
        float eta =
                (1 - pow(2.0 / upperBound, 1 - theta)) /
                (1 - d_zeta(theta, 2) / zetan);
        float u = randomFloat[i];
        float uz = u * zetan;
        if (uz < 1) {
            d_randomNumber[i] = 1;
        } else if (uz < 1 + pow(0.5, theta)) {
            d_randomNumber[i] = 2;
        } else {
            d_randomNumber[i] = 1 + (int) (upperBound * pow(eta * u - eta + 1, alpha));
        }

        // make sure we start from 0
        d_randomNumber[i] -= 1;
    }
}

void RandomGenerator::generateZipfian(uint32_t *h_randomNumber, uint64_t count, int upperBound,
                                      float theta, uint32_t seed, bool sorted) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Prepare random float
    curandGenerator_t gen;
    float *d_randomFloats;
    CUDA_CHECK(cudaMalloc((void **) &d_randomFloats, count * sizeof(float)));
    CUDA_CHECK(cudaStreamSynchronize(stream))
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetStream(gen,stream);

    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, d_randomFloats, count);


    // compute zetan
    dim3 dimBlockZetan(1024, 1, 1);
    dim3 dimGridZetan((upperBound + dimBlockZetan.x - 1) / dimBlockZetan.x, 1, 1);
    float *d_zetan;
    CUDA_CHECK(cudaMalloc(&d_zetan, sizeof(float)));
    cudaStreamSynchronize(stream);

    zetaKernel<<<dimGridZetan, dimBlockZetan, 0, stream>>>(upperBound, theta, d_zetan);
    cudaStreamSynchronize(stream);

    float *h_zetan;
    CUDA_CHECK(cudaMallocHost(&h_zetan, sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(h_zetan, d_zetan, sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // call the zipf kernel
    dim3 dimBlockZipf(1024, 1, 1);
    dim3 dimGridZipf((count + dimBlockZipf.x - 1) / dimBlockZipf.x, 1, 1);
    uint32_t *d_randomNumbers;
    CUDA_CHECK(cudaMallocHost(&d_randomNumbers, count * sizeof(uint32_t)));
    cudaStreamSynchronize(stream);
    zipfKernel<<<dimGridZipf, dimBlockZipf, 0, stream>>>(d_randomNumbers, count, upperBound, theta, *h_zetan,
                                                         d_randomFloats);
    cudaStreamSynchronize(stream);

    if (sorted) {
        /* Sort the generated floats */
        thrust::device_ptr<uint32_t> d_sortedRandomNumbers(d_randomNumbers);
        thrust::sort(d_sortedRandomNumbers, d_sortedRandomNumbers + count);

        CUDA_CHECK(cudaMemcpyAsync(h_randomNumber, thrust::raw_pointer_cast(d_sortedRandomNumbers),
                                   count * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, stream));

    } else {
        CUDA_CHECK(cudaMemcpyAsync(h_randomNumber, d_randomNumbers, count * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                   stream));
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    curandDestroyGenerator(gen);
    CUDA_CHECK(cudaFreeHost(h_zetan));
    CUDA_CHECK(cudaFree(d_randomFloats));
    CUDA_CHECK(cudaFreeHost(d_randomNumbers));
    CUDA_CHECK(cudaFree(d_zetan));
}

__global__ void generateSineDist(float *values, uint64_t count) {
    auto i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < count) {
        values[i] = sin((i % 360) * 3.14159 / 180) + 1;
    }
}

__global__ void generateProbability(float *out, float *in, float sum, uint64_t count) {
    auto i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < count) {
        out[i] = in[i] / sum;
    }
}

__device__
int64_t binary_search(const float *cumSum, float val, const uint64_t count) {
    uint64_t low = 0;
    uint64_t high = count - 1;
    uint64_t mid;
    while (high >= low) {
        mid = floor((low + high) / 2);
        if (cumSum[mid] < val) {
            low = mid + 1;
        } else if (cumSum[mid] > val) {
            high = mid - 1;
        } else {
            return mid;
        }
    }
    if ((low - high) == 1) {
        return min(low, high) + 1;
    } else {
        return -1;
    }

}

__global__ void searchSortedKernel(const float *cumSum, const uint64_t count, uint64_t *res, uint64_t upperBound) {
    uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < count) {
        res[i] = binary_search(cumSum, 1.0 * i / count, count) % upperBound;
    }
}

uint64_t *RandomGenerator::generateSineDistributedData(uint64_t count, uint32_t upperBound) {
    // define launch parameter
    dim3 dimBlock(1024, 1);
    dim3 dimGrid;
    dimGrid = (count + dimBlock.x - 1) / dimBlock.x;

    // generate base sine distribution (+1 to allow only positive)
    float *values;
    CUDA_CHECK(cudaHostAlloc(&values, count * sizeof(float), cudaHostAllocMapped));
    generateSineDist<<<dimGrid, dimBlock>>>(values, count);

    CUDA_CHECK(cudaDeviceSynchronize());

    // compute the sum (use atomic add)
    float sum;
    sum = thrust::reduce(values, values + count);
    CUDA_CHECK(cudaDeviceSynchronize());

    // generate the probability
    float *probs;
    CUDA_CHECK(cudaHostAlloc(&probs, count * sizeof(float), cudaHostAllocMapped));
    generateProbability<<<dimGrid, dimBlock>>>(probs, values, sum, count);
    CUDA_CHECK(cudaDeviceSynchronize());

    // compute cumulative sum
    float *cumSum;
    CUDA_CHECK(cudaHostAlloc(&cumSum, count * sizeof(float), cudaHostAllocMapped))
    thrust::exclusive_scan(probs, probs + count, cumSum); // in-place scan

    CUDA_CHECK(cudaDeviceSynchronize());

    // search the index
    uint64_t *pos;
    CUDA_CHECK(cudaHostAlloc(&pos, count * sizeof(uint64_t), cudaHostAllocMapped));
    searchSortedKernel<<<dimGrid, dimBlock>>>(cumSum, count, pos, upperBound);
    CUDA_CHECK(cudaDeviceSynchronize());

    // cleanup
    CUDA_CHECK(cudaFreeHost(values));
    CUDA_CHECK(cudaFreeHost(probs));
    CUDA_CHECK(cudaFreeHost(cumSum));

    thrust::sort(pos, pos + count);

    return pos; // possibly leaked, TODO: manage this outside
}