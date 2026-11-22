#pragma once

#include <cstddef>

namespace cuda_backend {

    enum class ActivationOp {
        Linear,
        ReLU,
        Sigmoid,
        Tanh
    };

    bool is_gpu_available();

    void* alloc_device(size_t bytes);
    void free_device(void* ptr) noexcept;
    void copy_h2d(void* dst_dev, const void* src_host, size_t bytes);
    void copy_d2h(void* dst_host, const void* src_dev, size_t bytes);
    void copy_d2d(void* dst_dev, const void* src_dev, size_t bytes);
    void memset_device_zero(void* dst_dev, size_t bytes);
    template <typename T>
    void memset_device_fill(void* dst_dev, size_t bytes, T val);

    template <typename T>
    class DeviceBuffer {
    private:
        T* ptr_ = nullptr;
        size_t capacity_ = 0;

    public:
        DeviceBuffer() = default;
        ~DeviceBuffer() { release(); }

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        DeviceBuffer(DeviceBuffer&& other) noexcept
            : ptr_(other.ptr_), capacity_(other.capacity_) {
            other.ptr_ = nullptr;
            other.capacity_ = 0;
        }

        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
            if (this != &other) {
                release();
                ptr_ = other.ptr_;
                capacity_ = other.capacity_;
                other.ptr_ = nullptr;
                other.capacity_ = 0;
            }
            return *this;
        }

        void ensure_capacity(size_t count) {
            if (count <= capacity_) return;
            release();
            if (count > 0) {
                ptr_ = static_cast<T*>(alloc_device(count * sizeof(T)));
                capacity_ = count;
            }
        }

        void release() noexcept {
            if (ptr_) {
                free_device(ptr_);
                ptr_ = nullptr;
                capacity_ = 0;
            }
        }

        [[nodiscard]] T* get() noexcept { return ptr_; }
        [[nodiscard]] const T* get() const noexcept { return ptr_; }
        [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
        [[nodiscard]] bool allocated() const noexcept { return ptr_ != nullptr; }
    };

    template <typename T>
    void multiply_matrices_device(
        const T* d_A, size_t A_height, size_t K, bool A_rowMajor,
        const T* d_B, size_t B_width, bool B_rowMajor,
        T* d_C, bool C_rowMajor
    );

    template <typename T>
    void sub_device(const T* d_A, const T* d_B, T* d_C, size_t count);

    template <typename T>
    void scale_device(const T* d_A, T scalar, T* d_C, size_t count);

    template <typename T>
    void hadamard_device(const T* d_A, const T* d_B, T* d_C, size_t count);

    template <typename T>
    void axpy_sub_inplace_device(T* d_W, const T* d_dW, T lr, size_t count);

    template <typename T>
    void add_horizontal_device(
        const T* d_A, const T* d_vec, T* d_C,
        size_t height, size_t width, bool rowMajor, T sign
    );

    template <typename T>
    void transpose_device(
        const T* d_in, T* d_out,
        size_t height, size_t width, bool rowMajor
    );

    template <typename T>
    void activation_device(const T* d_in, T* d_out, size_t count, ActivationOp op);

    template <typename T>
    void activation_deriv_device(const T* d_in, T* d_out, size_t count, ActivationOp op);

    template <typename T>
    void softmax_rows_device(
        const T* d_in, T* d_out,
        size_t height, size_t width, bool rowMajor
    );

    void clear_device_pool() noexcept;

    template <typename T>
    void mean_columns_device(
        const T* d_in, T* d_out,
        size_t height, size_t width, bool rowMajor
    );
}