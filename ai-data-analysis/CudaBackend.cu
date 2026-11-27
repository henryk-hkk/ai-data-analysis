#include "GpuBackend.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                     cudaGetErrorString(err));                \
        }                                                                     \
    } while (0)

namespace {

    std::atomic<bool> g_pool_alive{ true };

    class CudaMemoryPool {
    private:
        static constexpr size_t ALIGNMENT = 256;
        static constexpr size_t MAX_CACHED_BYTES = 512ULL * 1024 * 1024;

        std::mutex mtx_;
        std::unordered_map<size_t, std::vector<void*>> free_blocks_;
        std::unordered_map<void*, size_t> active_sizes_;
        size_t cached_bytes_ = 0;

        static size_t round_up(size_t bytes) noexcept {
            return (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        }

        void clear_free_locked() noexcept {
            for (auto& [bucket, ptrs] : free_blocks_) {
                for (void* ptr : ptrs) {
                    cudaFree(ptr);
                }
            }
            free_blocks_.clear();
            cached_bytes_ = 0;
        }

    public:
        ~CudaMemoryPool() {
            g_pool_alive.store(false, std::memory_order_release);
            clear();
        }

        void* allocate(size_t bytes) {
            if (bytes == 0) return nullptr;
            const size_t bucket = round_up(bytes);

            std::lock_guard<std::mutex> lock(mtx_);
            auto it = free_blocks_.find(bucket);
            if (it != free_blocks_.end() && !it->second.empty()) {
                void* ptr = it->second.back();
                it->second.pop_back();
                cached_bytes_ -= bucket;
                active_sizes_[ptr] = bucket;
                return ptr;
            }

            void* ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, bucket);
            if (err == cudaErrorMemoryAllocation) {
                cudaGetLastError();
                clear_free_locked();
                err = cudaMalloc(&ptr, bucket);
            }
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
            }

            active_sizes_[ptr] = bucket;
            return ptr;
        }

        void deallocate(void* ptr) noexcept {
            if (!ptr) return;

            std::lock_guard<std::mutex> lock(mtx_);
            auto it = active_sizes_.find(ptr);
            if (it == active_sizes_.end()) {
                cudaFree(ptr);
                return;
            }

            const size_t bucket = it->second;
            active_sizes_.erase(it);

            if (cached_bytes_ + bucket <= MAX_CACHED_BYTES) {
                free_blocks_[bucket].push_back(ptr);
                cached_bytes_ += bucket;
            }
            else {
                cudaFree(ptr);
            }
        }

        void clear() noexcept {
            std::lock_guard<std::mutex> lock(mtx_);
            clear_free_locked();
        }
    };

    CudaMemoryPool& get_memory_pool() {
        static CudaMemoryPool pool;
        return pool;
    }

    template <typename T>
    __global__ void fill_kernel(T* __restrict__ dst, size_t n, T val) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) dst[i] = val;
    }

    template <typename T, bool A_RM, bool B_RM, bool C_RM, int TILE>
    __global__ void matmul_tiled_kernel(
        const T* __restrict__ A, unsigned int M, unsigned int K,
        const T* __restrict__ B, unsigned int N,
        T* __restrict__ C
    ) {
        __shared__ T As[TILE][TILE + 1];
        __shared__ T Bs[TILE][TILE + 1];

        const unsigned int tx = threadIdx.x;
        const unsigned int ty = threadIdx.y;
        const unsigned int col = blockIdx.x * TILE + tx;
        const unsigned int row = blockIdx.y * TILE + ty;

        T sum = T{ 0 };
        const unsigned int numTiles = (K + TILE - 1) / TILE;

        for (unsigned int t = 0; t < numTiles; ++t) {
            if constexpr (A_RM) {
                const unsigned int aCol = t * TILE + tx;
                As[ty][tx] = (row < M && aCol < K)
                    ? A[static_cast<size_t>(row) * K + aCol]
                    : T{ 0 };
            }
            else {
                const unsigned int aRowT = blockIdx.y * TILE + tx;
                const unsigned int aColT = t * TILE + ty;
                As[tx][ty] = (aRowT < M && aColT < K)
                    ? A[static_cast<size_t>(aColT) * M + aRowT]
                    : T{ 0 };
            }

            if constexpr (B_RM) {
                const unsigned int bRow = t * TILE + ty;
                Bs[ty][tx] = (bRow < K && col < N)
                    ? B[static_cast<size_t>(bRow) * N + col]
                    : T{ 0 };
            }
            else {
                const unsigned int bRowT = t * TILE + tx;
                const unsigned int bColT = blockIdx.x * TILE + ty;
                Bs[tx][ty] = (bRowT < K && bColT < N)
                    ? B[static_cast<size_t>(bColT) * K + bRowT]
                    : T{ 0 };
            }

            __syncthreads();

#pragma unroll
            for (int k = 0; k < TILE; ++k) {
                sum += As[ty][k] * Bs[k][tx];
            }

            __syncthreads();
        }

        if (row < M && col < N) {
            if constexpr (C_RM)
                C[static_cast<size_t>(row) * N + col] = sum;
            else
                C[static_cast<size_t>(col) * M + row] = sum;
        }
    }

    template <typename T, bool A_RM, bool B_RM, bool C_RM, int TILE>
    void launch_matmul(
        dim3 gridDim, dim3 blockDim,
        const T* d_A, unsigned int M, unsigned int K,
        const T* d_B, unsigned int N,
        T* d_C
    ) {
        matmul_tiled_kernel<T, A_RM, B_RM, C_RM, TILE> << <gridDim, blockDim >> > (
            d_A, M, K, d_B, N, d_C
            );
    }

    template <typename T, int TILE>
    void dispatch_tiled_matmul(
        dim3 gridDim, dim3 blockDim,
        const T* d_A, unsigned int M, unsigned int K, bool A_rowMajor,
        const T* d_B, unsigned int N, bool B_rowMajor,
        T* d_C, bool C_rowMajor
    ) {
        const int layoutMask = (static_cast<int>(A_rowMajor) << 2)
            | (static_cast<int>(B_rowMajor) << 1)
            | static_cast<int>(C_rowMajor);

        switch (layoutMask) {
        case 0b000: launch_matmul<T, false, false, false, TILE>(gridDim, blockDim, d_A, M, K, d_B, N, d_C); break;
        case 0b001: launch_matmul<T, false, false, true, TILE>(gridDim, blockDim, d_A, M, K, d_B, N, d_C); break;
        case 0b010: launch_matmul<T, false, true, false, TILE>(gridDim, blockDim, d_A, M, K, d_B, N, d_C); break;
        case 0b011: launch_matmul<T, false, true, true, TILE>(gridDim, blockDim, d_A, M, K, d_B, N, d_C); break;
        case 0b100: launch_matmul<T, true, false, false, TILE>(gridDim, blockDim, d_A, M, K, d_B, N, d_C); break;
        case 0b101: launch_matmul<T, true, false, true, TILE>(gridDim, blockDim, d_A, M, K, d_B, N, d_C); break;
        case 0b110: launch_matmul<T, true, true, false, TILE>(gridDim, blockDim, d_A, M, K, d_B, N, d_C); break;
        case 0b111: launch_matmul<T, true, true, true, TILE>(gridDim, blockDim, d_A, M, K, d_B, N, d_C); break;
        }
    }

    template <typename T>
    __global__ void sub_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, size_t n) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) C[i] = A[i] - B[i];
    }

    template <typename T>
    __global__ void scale_kernel(const T* __restrict__ A, T scalar, T* __restrict__ C, size_t n) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) C[i] = A[i] * scalar;
    }

    template <typename T>
    __global__ void hadamard_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, size_t n) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) C[i] = A[i] * B[i];
    }

    template <typename T>
    __global__ void axpy_sub_inplace_kernel(T* __restrict__ W, const T* __restrict__ dW, T lr, size_t n) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) W[i] -= lr * dW[i];
    }

    template <typename T>
    __global__ void add_horizontal_kernel(
        const T* __restrict__ A, const T* __restrict__ vec, T* __restrict__ C,
        unsigned int height, unsigned int width, bool rowMajor, T sign
    ) {
        const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
        if (row < height && col < width) {
            const size_t idx = rowMajor
                ? (static_cast<size_t>(row) * width + col)
                : (static_cast<size_t>(col) * height + row);
            C[idx] = A[idx] + sign * vec[col];
        }
    }

    // transpozycja (RowMajor M x N -> N x M)
    template <typename T, int TILE>
    __global__ void transpose_tiled_kernel(
        const T* __restrict__ in, T* __restrict__ out,
        unsigned int M, unsigned int N
    ) {
        __shared__ T tile[TILE][TILE + 1];

        unsigned int col = blockIdx.x * TILE + threadIdx.x;
        unsigned int row = blockIdx.y * TILE + threadIdx.y;

        if (row < M && col < N) {
            tile[threadIdx.y][threadIdx.x] = in[static_cast<size_t>(row) * N + col];
        }

        __syncthreads();

        unsigned int outCol = blockIdx.y * TILE + threadIdx.x;
        unsigned int outRow = blockIdx.x * TILE + threadIdx.y;

        if (outRow < N && outCol < M) {
            out[static_cast<size_t>(outRow) * M + outCol] = tile[threadIdx.x][threadIdx.y];
        }
    }

    template <typename T>
    __global__ void activation_kernel(
        const T* __restrict__ in, T* __restrict__ out, size_t n, GpuBackend::ActivationOp op
    ) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n) return;

        const T x = in[i];
        switch (op) {
        case GpuBackend::ActivationOp::ReLU:
            out[i] = (x > T{ 0 }) ? x : T{ 0 };
            break;
        case GpuBackend::ActivationOp::Sigmoid:
            if constexpr (std::is_same_v<T, float>)
                out[i] = 1.0f / (1.0f + expf(-x));
            else
                out[i] = static_cast<T>(1.0 / (1.0 + exp(-static_cast<double>(x))));
            break;
        case GpuBackend::ActivationOp::Tanh:
            if constexpr (std::is_same_v<T, float>)
                out[i] = tanhf(x);
            else
                out[i] = static_cast<T>(tanh(static_cast<double>(x)));
            break;
        case GpuBackend::ActivationOp::Linear:
        default:
            out[i] = x;
            break;
        }
    }

    template <typename T>
    __global__ void activation_deriv_kernel(
        const T* __restrict__ a, T* __restrict__ out, size_t n, GpuBackend::ActivationOp op
    ) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n) return;

        const T val = a[i];
        switch (op) {
        case GpuBackend::ActivationOp::ReLU:
            out[i] = (val > T{ 0 }) ? T{ 1 } : T{ 0 };
            break;
        case GpuBackend::ActivationOp::Sigmoid:
            out[i] = val * (T{ 1 } - val);
            break;
        case GpuBackend::ActivationOp::Tanh:
            out[i] = T{ 1 } - val * val;
            break;
        case GpuBackend::ActivationOp::Linear:
        default:
            out[i] = T{ 1 };
            break;
        }
    }

    template <typename T>
    __global__ void softmax_rows_kernel(
        const T* __restrict__ in, T* __restrict__ out,
        unsigned int height, unsigned int width, bool rowMajor
    ) {
        const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= height || width == 0) return;

        auto idxAt = [&](unsigned int col) -> size_t {
            return rowMajor
                ? (static_cast<size_t>(row) * width + col)
                : (static_cast<size_t>(col) * height + row);
            };

        double maxVal = static_cast<double>(in[idxAt(0)]);
        for (unsigned int col = 1; col < width; ++col) {
            const double v = static_cast<double>(in[idxAt(col)]);
            if (v > maxVal) maxVal = v;
        }

        double sumExp = 0.0;
        for (unsigned int col = 0; col < width; ++col) {
            const double e = exp(static_cast<double>(in[idxAt(col)]) - maxVal);
            out[idxAt(col)] = static_cast<T>(e);
            sumExp += e;
        }

        const double invSum = 1.0 / sumExp;
        for (unsigned int col = 0; col < width; ++col) {
            out[idxAt(col)] = static_cast<T>(static_cast<double>(out[idxAt(col)]) * invSum);
        }
    }

    template <typename T>
    __global__ void mean_columns_kernel(
        const T* __restrict__ in, T* __restrict__ out,
        unsigned int height, unsigned int width, bool rowMajor
    ) {
        const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= width) return;

        double sum = 0.0;
        for (unsigned int row = 0; row < height; ++row) {
            const size_t idx = rowMajor
                ? (static_cast<size_t>(row) * width + col)
                : (static_cast<size_t>(col) * height + row);
            sum += static_cast<double>(in[idx]);
        }
        out[col] = static_cast<T>(sum / static_cast<double>(height));
    }

    inline unsigned int grid1D(size_t count, unsigned int threads = 256) {
        return static_cast<unsigned int>((count + threads - 1) / threads);
    }

} // namespace

bool GpuBackend::deviceAvailable() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess && deviceCount > 0);
}

void* GpuBackend::alloc(size_t bytes) {
    if (bytes == 0) return nullptr;
    return get_memory_pool().allocate(bytes);
}

void GpuBackend::free(void* ptr) noexcept {
    if (!ptr) return;
    if (g_pool_alive.load(std::memory_order_acquire)) {
        get_memory_pool().deallocate(ptr);
    }
    else {
        cudaFree(ptr);
    }
}

void GpuBackend::clear_device_pool() noexcept {
    if (g_pool_alive.load(std::memory_order_acquire)) {
        get_memory_pool().clear();
    }
}

void GpuBackend::copy_Host2Device(void* dst_dev, const void* src_host, size_t bytes) {
    if (bytes > 0) {
        CUDA_CHECK(cudaMemcpy(dst_dev, src_host, bytes, cudaMemcpyHostToDevice));
    }
}

void GpuBackend::copy_Device2Host(void* dst_host, const void* src_dev, size_t bytes) {
    if (bytes > 0) {
        CUDA_CHECK(cudaMemcpy(dst_host, src_dev, bytes, cudaMemcpyDeviceToHost));
    }
}

void GpuBackend::copy_Device2Device(void* dst_dev, const void* src_dev, size_t bytes) {
    if (bytes > 0) {
        CUDA_CHECK(cudaMemcpy(dst_dev, src_dev, bytes, cudaMemcpyDeviceToDevice));
    }
}

void GpuBackend::zero(void* dst_dev, size_t bytes) {
    if (bytes > 0) {
        CUDA_CHECK(cudaMemset(dst_dev, 0, bytes));
    }
}

template <gpu_compatible T>
void GpuBackend::fill(T* dst_dev, size_t count, T val) {
    if (count == 0) return;
    fill_kernel<T> << <grid1D(count), 256 >> > (dst_dev, count, val);
    CUDA_CHECK(cudaGetLastError());
}

template <gpu_compatible T>
void GpuBackend::matmul(
    const T* d_A, size_t A_height, size_t K, bool A_rowMajor,
    const T* d_B, size_t B_width, bool B_rowMajor,
    T* d_C, bool C_rowMajor
) {
    const size_t sizeC = A_height * B_width;
    if (sizeC == 0) return;

    if (K == 0) {
        GpuBackend::zero(d_C, sizeC * sizeof(T));
        return;
    }

    constexpr size_t maxDim = std::numeric_limits<unsigned int>::max();
    if (A_height > maxDim || K > maxDim || B_width > maxDim) {
        throw std::overflow_error("Matrix dimensions exceed 32-bit CUDA kernel indexing limit");
    }

    constexpr int TILE = 16;
    dim3 blockDim(TILE, TILE);
    dim3 gridDim(
        static_cast<unsigned int>((B_width + TILE - 1) / TILE),
        static_cast<unsigned int>((A_height + TILE - 1) / TILE)
    );

    dispatch_tiled_matmul<T, TILE>(
        gridDim, blockDim,
        d_A, static_cast<unsigned int>(A_height), static_cast<unsigned int>(K), A_rowMajor,
        d_B, static_cast<unsigned int>(B_width), B_rowMajor,
        d_C, C_rowMajor
    );
    CUDA_CHECK(cudaGetLastError());
}

template <gpu_compatible T>
void GpuBackend::subtract(const T* d_A, const T* d_B, T* d_C, size_t count) {
    if (count == 0) return;
    sub_kernel<T> << <grid1D(count), 256 >> > (d_A, d_B, d_C, count);
    CUDA_CHECK(cudaGetLastError());
}

template <gpu_compatible T>
void GpuBackend::scale(const T* d_A, T scalar, T* d_C, size_t count) {
    if (count == 0) return;
    scale_kernel<T> << <grid1D(count), 256 >> > (d_A, scalar, d_C, count);
    CUDA_CHECK(cudaGetLastError());
}

template <gpu_compatible T>
void GpuBackend::hadamard(const T* d_A, const T* d_B, T* d_C, size_t count) {
    if (count == 0) return;
    hadamard_kernel<T> << <grid1D(count), 256 >> > (d_A, d_B, d_C, count);
    CUDA_CHECK(cudaGetLastError());
}

template <gpu_compatible T>
void GpuBackend::axpy_subtract(T* d_W, const T* d_dW, T lr, size_t count) {
    if (count == 0) return;
    axpy_sub_inplace_kernel<T> << <grid1D(count), 256 >> > (d_W, d_dW, lr, count);
    CUDA_CHECK(cudaGetLastError());
}

template <gpu_compatible T>
void GpuBackend::add_horizontal(
    const T* d_A, const T* d_vec, T* d_C,
    size_t height, size_t width, bool rowMajor, T sign
) {
    if (height == 0 || width == 0) return;
    dim3 blockDim(16, 16);
    dim3 gridDim(
        static_cast<unsigned int>((width + 15) / 16),
        static_cast<unsigned int>((height + 15) / 16)
    );
    add_horizontal_kernel<T> << <gridDim, blockDim >> > (
        d_A, d_vec, d_C,
        static_cast<unsigned int>(height), static_cast<unsigned int>(width),
        rowMajor, sign
        );
    CUDA_CHECK(cudaGetLastError());
}

template <gpu_compatible T>
void GpuBackend::transpose(
    const T* d_in, T* d_out,
    size_t height, size_t width, bool rowMajor
) {
    if (height == 0 || width == 0) return;
    constexpr int TILE = 16;
    const unsigned int M = static_cast<unsigned int>(rowMajor ? height : width);
    const unsigned int N = static_cast<unsigned int>(rowMajor ? width : height);
    dim3 blockDim(TILE, TILE);
    dim3 gridDim((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    transpose_tiled_kernel<T, TILE> << <gridDim, blockDim >> > (d_in, d_out, M, N);
    CUDA_CHECK(cudaGetLastError());
}

template <gpu_compatible T>
void GpuBackend::activation(const T* d_in, T* d_out, size_t count, ActivationOp op) {
    if (count == 0) return;
    activation_kernel<T> << <grid1D(count), 256 >> > (d_in, d_out, count, op);
    CUDA_CHECK(cudaGetLastError());
}

template <gpu_compatible T>
void GpuBackend::activation_deriv(const T* d_in, T* d_out, size_t count, ActivationOp op) {
    if (count == 0) return;
    activation_deriv_kernel<T> << <grid1D(count), 256 >> > (d_in, d_out, count, op);
    CUDA_CHECK(cudaGetLastError());
}

template <gpu_compatible T>
void GpuBackend::softmax_rows(
    const T* d_in, T* d_out,
    size_t height, size_t width, bool rowMajor
) {
    if (height == 0 || width == 0) return;
    softmax_rows_kernel<T> << <grid1D(height, 128), 128 >> > (
        d_in, d_out,
        static_cast<unsigned int>(height), static_cast<unsigned int>(width), rowMajor
        );
    CUDA_CHECK(cudaGetLastError());
}

template <gpu_compatible T>
void GpuBackend::mean_columns(
    const T* d_in, T* d_out,
    size_t height, size_t width, bool rowMajor
) {
    if (height == 0 || width == 0) return;
    mean_columns_kernel<T> << <grid1D(width, 128), 128 >> > (
        d_in, d_out,
        static_cast<unsigned int>(height), static_cast<unsigned int>(width), rowMajor
        );
    CUDA_CHECK(cudaGetLastError());
}

#define INSTANTIATE_CUDA_OPS(T)                                                                               \
    template void GpuBackend::fill<T>(T*, size_t, T);                                                         \
    template void GpuBackend::matmul<T>(const T*, size_t, size_t, bool, const T*, size_t, bool, T*, bool);    \
    template void GpuBackend::subtract<T>(const T*, const T*, T*, size_t);                                    \
    template void GpuBackend::scale<T>(const T*, T, T*, size_t);                                              \
    template void GpuBackend::hadamard<T>(const T*, const T*, T*, size_t);                                    \
    template void GpuBackend::axpy_subtract<T>(T*, const T*, T, size_t);                                      \
    template void GpuBackend::add_horizontal<T>(const T*, const T*, T*, size_t, size_t, bool, T);             \
    template void GpuBackend::transpose<T>(const T*, T*, size_t, size_t, bool);                               \
    template void GpuBackend::activation<T>(const T*, T*, size_t, GpuBackend::ActivationOp);                  \
    template void GpuBackend::activation_deriv<T>(const T*, T*, size_t, GpuBackend::ActivationOp);            \
    template void GpuBackend::softmax_rows<T>(const T*, T*, size_t, size_t, bool);                            \
    template void GpuBackend::mean_columns<T>(const T*, T*, size_t, size_t, bool);

INSTANTIATE_CUDA_OPS(float)
INSTANTIATE_CUDA_OPS(double)
INSTANTIATE_CUDA_OPS(int)

#undef INSTANTIATE_CUDA_OPS