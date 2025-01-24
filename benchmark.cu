#include <unistd.h>

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cassert>
#include <curand.h>
#include <random>
#include "cuda_runtime_api.h"
#include <cub/block/block_load.cuh>
#include <cub/util_type.cuh>

using namespace cub;

#define CURAND_CALL(x)                                                     \
    do                                                                     \
    {                                                                      \
        curandStatus ret = (x);                                            \
        if (ret != CURAND_STATUS_SUCCESS)                                  \
        {                                                                  \
            printf("cuRAND Error %d at %s:%d\n", ret, __FILE__, __LINE__); \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

class CudaException : public std::runtime_error
{
public:
    explicit CudaException(const std::string &what) : runtime_error(what) {}
};

inline void cuda_check_(cudaError_t val, const char *file, int line)
{
    if (val != cudaSuccess)
    {
        throw CudaException(std::string(file) + ":" + std::to_string(line) + ": CUDA error " + std::to_string(val) + ": " + cudaGetErrorString(val));
    }
}

inline void cuda_check_last_error_(const char *file, int line)
{
    cudaDeviceSynchronize();
    cudaError_t err = cudaPeekAtLastError();
    cuda_check_(err, file, line);
}

#define CUDA_CHECK(val)                         \
    {                                           \
        cuda_check_((val), __FILE__, __LINE__); \
    }
#define CUDA_CHECK_LAST_ERROR()                     \
    {                                               \
        cuda_check_last_error_(__FILE__, __LINE__); \
    }

template <typename InputIterator,
          typename OutputIterator,
          int BLOCK_THREADS,
          int ITEMS_PER_THREAD>
struct DeviceExample
{
    using KeyInT = detail::value_t<InputIterator>;
    using BlockLoadInputT = BlockLoad<KeyInT, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_VECTORIZE>;
    static constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

    InputIterator in;
    OutputIterator out;
    int total_num;

    __device__ DeviceExample(InputIterator in, OutputIterator out, int total_num)
        : in(in), out(out), total_num(total_num)
    {
        assert(total_num == BLOCK_THREADS * ITEMS_PER_THREAD);
    }
    __device__ __forceinline__ void VectorizedProcess()
    {
        KeyInT thread_data[ITEMS_PER_THREAD];
        int tile_base = blockIdx.x * TILE_ITEMS;
        int offset = threadIdx.x * ITEMS_PER_THREAD + tile_base;

        __shared__ typename BlockLoadInputT::TempStorage temp_storage;
        BlockLoadInputT(temp_storage).Load(in + tile_base, thread_data);

        for (int i = 0; i < ITEMS_PER_THREAD; i++)
        {
            out[offset + i] = thread_data[i] * 2;
        }
    }
};

template <typename InputIterator,
          typename OutputIterator,
          int BLOCK_THREADS,
          int ITEMS_PER_THREAD>
__global__ void testBlockLoad(InputIterator in, OutputIterator out, int total_num)
{
    using DeviceExampleT = DeviceExample<InputIterator, OutputIterator, BLOCK_THREADS, ITEMS_PER_THREAD>;

    DeviceExampleT(in, out, total_num).VectorizedProcess();
}

template <typename T,
          int BLOCK_THREADS,
          int ITEMS_PER_THREAD>
struct DeviceExampleTwo
{
    using BlockLoadInputT = BlockLoad<T, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_VECTORIZE>;
    static constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

    T *in;
    T *out;
    int total_num;

    __device__ DeviceExampleTwo(T *in, T *out, int total_num)
        : in(in), out(out), total_num(total_num)
    {
        assert(total_num == BLOCK_THREADS * ITEMS_PER_THREAD);
    }
    __device__ __forceinline__ void VectorizedProcess()
    {
        T thread_data[ITEMS_PER_THREAD];
        int tile_base = blockIdx.x * TILE_ITEMS;
        int offset = threadIdx.x * ITEMS_PER_THREAD + tile_base;

        __shared__ typename BlockLoadInputT::TempStorage temp_storage;
        BlockLoadInputT(temp_storage).Load(in + tile_base, thread_data);

        for (int i = 0; i < ITEMS_PER_THREAD; i++)
        {
            out[offset + i] = thread_data[i] * 2;
        }
    }
};

template <typename T,
          int BLOCK_THREADS,
          int ITEMS_PER_THREAD>
__global__ void testBlockLoadTwo(T *in, T *out, int total_num)
{
    using DeviceExampleT = DeviceExampleTwo<T, BLOCK_THREADS, ITEMS_PER_THREAD>;

    DeviceExampleT(in, out, total_num).VectorizedProcess();
}

template <typename T,
          int BLOCK_THREADS,
          int ITEMS_PER_THREAD>
struct DeviceExampleConst
{
    using BlockLoadInputT = BlockLoad<T, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_VECTORIZE>;
    static constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

    const T *in;
    T *out;
    int total_num;

    __device__ DeviceExampleConst(const T *in, T *out, int total_num)
        : in(in), out(out), total_num(total_num)
    {
        assert(total_num == BLOCK_THREADS * ITEMS_PER_THREAD);
    }
    __device__ __forceinline__ void VectorizedProcess()
    {
        T thread_data[ITEMS_PER_THREAD];
        int tile_base = blockIdx.x * TILE_ITEMS;
        int offset = threadIdx.x * ITEMS_PER_THREAD + tile_base;

        __shared__ typename BlockLoadInputT::TempStorage temp_storage;
        BlockLoadInputT(temp_storage).Load(in + tile_base, thread_data);

        for (int i = 0; i < ITEMS_PER_THREAD; i++)
        {
            out[offset + i] = thread_data[i] * 2;
        }
    }
};

template <typename T,
          int BLOCK_THREADS,
          int ITEMS_PER_THREAD>
__global__ void testBlockLoadConst(const T *in, T *out, int total_num)
{
    using DeviceExampleT = DeviceExampleConst<T, BLOCK_THREADS, ITEMS_PER_THREAD>;

    DeviceExampleT(in, out, total_num).VectorizedProcess();
}

int main(int argc, char **argv)
{
    constexpr int BLOCK_NUM = 10;
    constexpr int BLOCK_THREADS = 256;
    constexpr int ITEMS_PER_THREAD = 4;

    int size = ITEMS_PER_THREAD * BLOCK_THREADS * BLOCK_NUM;
    int size_in_bytes = size * sizeof(float);

    bool run_extra_tests = false;

    float *d_in = nullptr;
    float *d_out = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&d_in, size_in_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, size_in_bytes));

    curandGenerator_t gen_;
    CURAND_CALL(curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen_, std::random_device{}()));
    CURAND_CALL(curandGenerateUniform(gen_, d_in, size));

    // Tests for pointer-to-const
    using InputIterator = float *;
    using OutputIterator = float *;
    testBlockLoad<InputIterator, OutputIterator, BLOCK_THREADS, ITEMS_PER_THREAD><<<BLOCK_NUM, BLOCK_THREADS>>>(d_in, d_out, size);
    CUDA_CHECK_LAST_ERROR();

    if (run_extra_tests)
    {
        testBlockLoadTwo<float, BLOCK_THREADS, ITEMS_PER_THREAD><<<BLOCK_NUM, BLOCK_THREADS>>>(d_in, d_out, size);
        CUDA_CHECK_LAST_ERROR();

        // Tests for pointer-to-const
        testBlockLoadConst<float, BLOCK_THREADS, ITEMS_PER_THREAD><<<BLOCK_NUM, BLOCK_THREADS>>>(d_in, d_out, size);
        CUDA_CHECK_LAST_ERROR();

        using InputConstIterator = const float *;
        testBlockLoad<InputConstIterator, OutputIterator, BLOCK_THREADS, ITEMS_PER_THREAD><<<BLOCK_NUM, BLOCK_THREADS>>>(d_in, d_out, size);
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CURAND_CALL(curandDestroyGenerator(gen_));
}
