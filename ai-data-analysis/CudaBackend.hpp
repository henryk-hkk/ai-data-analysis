#pragma once
#include <cstddef>

namespace cuda_backend {

    bool is_gpu_available();

    template <typename T>
    void multiply_matrices(
        const T* h_A, size_t A_height, size_t K, bool A_rowMajor,
        const T* h_B, size_t B_width, bool B_rowMajor,
        T* h_C, bool C_rowMajor
    );

}