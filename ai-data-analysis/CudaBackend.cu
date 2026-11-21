#include "CudaBackend.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
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
                // zamiana tx/ty wymusza ciagly odczyt po kolumnie w vramie
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
    void launch_kernel(
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
        case 0b000: launch_kernel<T, false, false, false, TILE>(gridDim, blockDim, d_A, M, K, d_B, N, d_C); break;
        case 0b001: launch_kernel<T, false, false, true, TILE>(gridDim, blockDim, d_A, M, K, d_B, N, d_C); break;
        case 0b010: launch_kernel<T, false, true, false, TILE>(gridDim, blockDim, d_A, M, K, d_B, N, d_C); break;
        case 0b011: launch_kernel<T, false, true, true, TILE>(gridDim, blockDim, d_A, M, K, d_B, N, d_C); break;
        case 0b100: launch_kernel<T, true, false, false, TILE>(gridDim, blockDim, d_A, M, K, d_B, N, d_C); break;
        case 0b101: launch_kernel<T, true, false, true, TILE>(gridDim, blockDim, d_A, M, K, d_B, N, d_C); break;
        case 0b110: launch_kernel<T, true, true, false, TILE>(gridDim, blockDim, d_A, M, K, d_B, N, d_C); break;
        case 0b111: launch_kernel<T, true, true, true, TILE>(gridDim, blockDim, d_A, M, K, d_B, N, d_C); break;
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
        const T* h_B, size_t B_width, bool B_rowMajor,
        T* h_C, bool C_rowMajor
    ) {
        const size_t sizeC = A_height * B_width;
        if (sizeC == 0) return;

        if (K == 0) {
            std::fill_n(h_C, sizeC, T{ 0 });
            return;
        }

        constexpr size_t maxDim = std::numeric_limits<unsigned int>::max();
        if (A_height > maxDim || K > maxDim || B_width > maxDim) {
            throw std::overflow_error("Matrix dimensions exceed 32-bit CUDA kernel indexing limit");
        }

        const size_t sizeA = A_height * K;
        const size_t sizeB = K * B_width;

        CudaBuffer<T> d_A(sizeA);
        CudaBuffer<T> d_B(sizeB);
        CudaBuffer<T> d_C(sizeC);

        CUDA_CHECK(cudaMemcpy(d_A.ptr, h_A, sizeA * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B.ptr, h_B, sizeB * sizeof(T), cudaMemcpyHostToDevice));

        constexpr int TILE = 16;
        dim3 blockDim(TILE, TILE);
        dim3 gridDim(
            static_cast<unsigned int>((B_width + TILE - 1) / TILE),
            static_cast<unsigned int>((A_height + TILE - 1) / TILE)
        );

        dispatch_tiled_matmul<T, TILE>(
            gridDim, blockDim,
            d_A.ptr, static_cast<unsigned int>(A_height), static_cast<unsigned int>(K), A_rowMajor,
            d_B.ptr, static_cast<unsigned int>(B_width), B_rowMajor,
            d_C.ptr, C_rowMajor
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_C, d_C.ptr, sizeC * sizeof(T), cudaMemcpyDeviceToHost));
    }

    template void multiply_matrices<float>(const float*, size_t, size_t, bool, const float*, size_t, bool, float*, bool);
    template void multiply_matrices<double>(const double*, size_t, size_t, bool, const double*, size_t, bool, double*, bool);
    template void multiply_matrices<int>(const int*, size_t, size_t, bool, const int*, size_t, bool, int*, bool);

}