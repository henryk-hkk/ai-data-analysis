#include "CudaBackend.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                     cudaGetErrorString(err));                \
        }                                                                     \
    } while (0)

namespace {

    template <typename T>
    struct CudaBuffer {
        T* ptr = nullptr;

        explicit CudaBuffer(size_t count) {
            if (count > 0) {
                CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
            }
        }
        ~CudaBuffer() {
            if (ptr) cudaFree(ptr);
        }
        CudaBuffer(const CudaBuffer&) = delete;
        CudaBuffer& operator=(const CudaBuffer&) = delete;
    };

    __device__ inline size_t get_idx(size_t x, size_t y, size_t width, size_t height, bool rowMajor) {
        return rowMajor ? (y * width + x) : (x * height + y);
    }

    template <typename T>
    __global__ void matmul_kernel(
        const T* __restrict__ A, size_t A_height, size_t K, bool A_rowMajor,
        const T* __restrict__ B, size_t B_width,            bool B_rowMajor,
        T*       __restrict__ C,                            bool C_rowMajor
    ) {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= B_width || y >= A_height) {
            return;
        }

        T sum = T{0};
        for (size_t k = 0; k < K; ++k) {
            T a_val = A[get_idx(k, y, K, A_height, A_rowMajor)];
            T b_val = B[get_idx(x, k, B_width, K, B_rowMajor)];
            sum += a_val * b_val;
        }

        C[get_idx(x, y, B_width, A_height, C_rowMajor)] = sum;
    }

    template <typename T, bool A_RM, bool B_RM, bool C_RM, int TILE = 16>
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
            const unsigned int aCol = t * TILE + tx;
            const unsigned int bRow = t * TILE + ty;

            if (row < M && aCol < K) {
                if constexpr (A_RM)
                    As[ty][tx] = A[static_cast<size_t>(row) * K + aCol];
                else
                    As[ty][tx] = A[static_cast<size_t>(aCol) * M + row];
            }
            else {
                As[ty][tx] = T{ 0 };
            }

            if (bRow < K && col < N) {
                if constexpr (B_RM)
                    Bs[ty][tx] = B[static_cast<size_t>(bRow) * N + col];
                else
                    Bs[ty][tx] = B[static_cast<size_t>(col) * K + bRow];
            }
            else {
                Bs[ty][tx] = T{ 0 };
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

}

namespace cuda_backend {

    bool is_gpu_available() {
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        return (err == cudaSuccess && deviceCount > 0);
    }

    template <typename T>
    void multiply_matrices(
        const T* h_A, size_t A_height, size_t K, bool A_rowMajor,
        const T* h_B, size_t B_width,            bool B_rowMajor,
        T*       h_C,                            bool C_rowMajor
    ) {
        const size_t sizeA = A_height * K;
        const size_t sizeB = K * B_width;
        const size_t sizeC = A_height * B_width;

        if (sizeC == 0) return;

        CudaBuffer<T> d_A(sizeA);
        CudaBuffer<T> d_B(sizeB);
        CudaBuffer<T> d_C(sizeC);

        CUDA_CHECK(cudaMemcpy(d_A.ptr, h_A, sizeA * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B.ptr, h_B, sizeB * sizeof(T), cudaMemcpyHostToDevice));

        dim3 blockDim(16, 16);
        dim3 gridDim(
            static_cast<unsigned int>((B_width  + blockDim.x - 1) / blockDim.x),
            static_cast<unsigned int>((A_height + blockDim.y - 1) / blockDim.y)
        );

        matmul_kernel<T><<<gridDim, blockDim>>>(
            d_A.ptr, A_height, K, A_rowMajor,
            d_B.ptr, B_width,     B_rowMajor,
            d_C.ptr,              C_rowMajor
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_C, d_C.ptr, sizeC * sizeof(T), cudaMemcpyDeviceToHost));

    }

template void multiply_matrices<float>(const float*, size_t, size_t, bool, const float*, size_t, bool, float*, bool);
template void multiply_matrices<double>(const double*, size_t, size_t, bool, const double*, size_t, bool, double*, bool);
template void multiply_matrices<int>(const int*, size_t, size_t, bool, const int*, size_t, bool, int*, bool);

}