#include "GpuBackend.hpp"
#include <stdexcept>
#include <string>

namespace {

    [[noreturn]] void throw_no_gpu(const char* op) {
        throw std::runtime_error(
            std::string("GpuBackend::") + op +
            "() wywolane w buildzie bez aktywnego backendu GPU (NullBackend)."
        );
    }

} // namespace

bool GpuBackend::deviceAvailable() {
    return false;
}

void* GpuBackend::alloc(size_t bytes) {
    if (bytes == 0) return nullptr;
    throw_no_gpu("alloc");
}

void GpuBackend::free(void*) noexcept {}

void GpuBackend::clear_device_pool() noexcept {}

void GpuBackend::copy_Host2Device(void*, const void*, size_t) {
    throw_no_gpu("copy_Host2Device");
}

void GpuBackend::copy_Device2Host(void*, const void*, size_t) {
    throw_no_gpu("copy_Device2Host");
}

void GpuBackend::copy_Device2Device(void*, const void*, size_t) {
    throw_no_gpu("copy_Device2Device");
}

void GpuBackend::zero(void*, size_t) {
    throw_no_gpu("zero");
}

template <gpu_compatible T>
void GpuBackend::fill(T*, size_t, T) {
    throw_no_gpu("fill");
}

template <gpu_compatible T>
void GpuBackend::matmul(
    const T*, size_t, size_t, bool,
    const T*, size_t, bool,
    T*, bool
) {
    throw_no_gpu("matmul");
}

template <gpu_compatible T>
void GpuBackend::subtract(const T*, const T*, T*, size_t) {
    throw_no_gpu("subtract");
}

template <gpu_compatible T>
void GpuBackend::scale(const T*, T, T*, size_t) {
    throw_no_gpu("scale");
}

template <gpu_compatible T>
void GpuBackend::hadamard(const T*, const T*, T*, size_t) {
    throw_no_gpu("hadamard");
}

template <gpu_compatible T>
void GpuBackend::axpy_subtract(T*, const T*, T, size_t) {
    throw_no_gpu("axpy_subtract");
}

template <gpu_compatible T>
void GpuBackend::add_horizontal(
    const T*, const T*, T*,
    size_t, size_t, bool, T
) {
    throw_no_gpu("add_horizontal");
}

template <gpu_compatible T>
void GpuBackend::transpose(
    const T*, T*,
    size_t, size_t, bool
) {
    throw_no_gpu("transpose");
}

template <gpu_compatible T>
void GpuBackend::activation(const T*, T*, size_t, ActivationOp) {
    throw_no_gpu("activation");
}

template <gpu_compatible T>
void GpuBackend::activation_deriv(const T*, T*, size_t, ActivationOp) {
    throw_no_gpu("activation_deriv");
}

template <gpu_compatible T>
void GpuBackend::softmax_rows(
    const T*, T*,
    size_t, size_t, bool
) {
    throw_no_gpu("softmax_rows");
}

template <gpu_compatible T>
void GpuBackend::mean_columns(
    const T*, T*,
    size_t, size_t, bool
) {
    throw_no_gpu("mean_columns");
}

#define INSTANTIATE_NULL_OPS(T)                                                                               \
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

INSTANTIATE_NULL_OPS(float)
INSTANTIATE_NULL_OPS(double)
INSTANTIATE_NULL_OPS(int)

#undef INSTANTIATE_NULL_OPS