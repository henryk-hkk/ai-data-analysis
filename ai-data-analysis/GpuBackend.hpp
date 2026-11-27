#pragma once
#include "Concepts.h"
class GpuBackend {
public:
    GpuBackend() = delete;

    enum class ActivationOp : uint8_t { Linear, ReLU, Sigmoid, Tanh };
    //enum class Layout : uint8_t { RowMajor, ColMajor };

    [[nodiscard]] static bool deviceAvailable();

    [[nodiscard]] static void* alloc(size_t bytes);
    static void free(void* ptr) noexcept;
    static void clear_device_pool() noexcept;

    static void copy_Host2Device(void* dst_dev, const void* src_host, size_t bytes);
    static void copy_Device2Host(void* dst_host, const void* src_dev, size_t bytes);
    static void copy_Device2Device(void* dst_dev, const void* src_dev, size_t bytes);


    template <gpu_compatible T>
    static void fill(T* dst_dev, size_t count, T val);
    static void zero(void* dst_dev, size_t bytes);

    template <gpu_compatible T>
    static void matmul(
        const T* d_A, size_t A_height, size_t K, bool A_rowMajor,
        const T* d_B, size_t B_width, bool B_rowMajor,
        T* d_C, bool C_rowMajor
    );

    template <gpu_compatible T>
    static void subtract(const T* d_A, const T* d_B, T* d_C, size_t count);

    template <gpu_compatible T>
    static void scale(const T* d_A, T scalar, T* d_C, size_t count);

    template <gpu_compatible T>
    static void hadamard(const T* d_A, const T* d_B, T* d_C, size_t count);

    template <gpu_compatible T>
    static void axpy_subtract(T* d_W, const T* d_dW, T lr, size_t count);

    template <gpu_compatible T>
    static void add_horizontal(
        const T* d_A, const T* d_vec, T* d_C,
        size_t height, size_t width, bool rowMajor, T sign
    );

    template <gpu_compatible T>
    static void transpose(
        const T* d_in, T* d_out,
        size_t height, size_t width, bool rowMajor
    );

    template <gpu_compatible T>
    static void activation(const T* d_in, T* d_out, size_t count, ActivationOp op);

    template <gpu_compatible T>
    static void activation_deriv(const T* d_in, T* d_out, size_t count, ActivationOp op);

    template <gpu_compatible T>
    static void softmax_rows(
        const T* d_in, T* d_out,
        size_t height, size_t width, bool rowMajor
    );

    template <gpu_compatible T>
    static void mean_columns(
        const T* d_in, T* d_out,
        size_t height, size_t width, bool rowMajor
    );

    template <gpu_compatible T>
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

        void ensureCapacity(size_t count) {
            if (count <= capacity_) return;
            release();
            if (count > 0) {
                ptr_ = static_cast<T*>(GpuBackend::alloc(count * sizeof(T)));
                capacity_ = count;
            }
        }

        void release() noexcept {
            if (ptr_) {
                GpuBackend::free(ptr_);
                ptr_ = nullptr;
                capacity_ = 0;
            }
        }

        [[nodiscard]] T* get() noexcept { return ptr_; }
        [[nodiscard]] const T* get() const noexcept { return ptr_; }
        [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
        [[nodiscard]] bool allocated() const noexcept { return ptr_ != nullptr; }
    };
};