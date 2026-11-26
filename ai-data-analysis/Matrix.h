#pragma once

#include <iostream>
#include <concepts>
#include <vector>
#include <stdexcept>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <ios>
#include <type_traits>
#include <algorithm>
#include <execution>
#include <ranges>
#include <cmath>
#include "Concepts.h"
#include "MatrixLayout.h"
#include "Dimensions.h"
#include "CudaBackend.hpp"

template<typename T = double, MatrixLayout L = MatrixLayout::RowMajor>
    requires arithmetic<T>
class Matrix {
private:
    size_t width = 0, height = 0;
    mutable std::vector<T> data;
    mutable cuda_backend::DeviceBuffer<T> d_buf;

    mutable bool hostDirty = false;   // VRAM ma nowsze dane niż RAM
    mutable bool deviceDirty = true;  // RAM ma nowsze dane niż VRAM

    static constexpr size_t DEFAULT_PRINT_PRECISION = 6;

    static constexpr bool supports_gpu =
        std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, int>;

    constexpr size_t index(size_t x, size_t y) const {
        if constexpr (L == MatrixLayout::RowMajor)
            return y * width + x;
        else
            return x * height + y;
    }

    template<typename U, MatrixLayout L2>
        requires arithmetic<U>
    friend class Matrix;

    static bool gpuAvailable() {
        if constexpr (supports_gpu) {
            static const bool hw_available = cuda_backend::is_gpu_available();
            return hw_available && cuda_backend::is_gpu_available();
        }
        return false;
    }
    static Matrix<T, L> createOnDevice(size_t rows, size_t cols) { // Allocates new Matrix on the GPU
        Matrix<T, L> m;
        m.height = rows;
        m.width = cols;
        const size_t n = rows * cols;
        if (n > 0) {
            m.d_buf.ensure_capacity(n);
            m.hostDirty = true;
            m.deviceDirty = false;
        }
        return m;
    }
public:
    // constructors, deconstructors, = operator

    Matrix() = default;
    Matrix(size_t rows, size_t cols)
        : width(cols), height(rows), data(width* height, T{}) {
    }
    Matrix(size_t rows, size_t cols, std::vector<T> content)
        : width(cols), height(rows), data(std::move(content)) {
        if (data.size() != width * height)
            throw std::invalid_argument("Vector size does not match matrix dimensions.");
    }
    Matrix(const std::vector<std::vector<T>>& content) {
        if (content.empty()) {
            width = 0;
            height = 0;
            return;
        }

        height = content.size();
        width = content[0].size();

        for (const auto& row : content) {
            if (row.size() != width) {
                throw std::invalid_argument("All rows in 2D vector must have the same length.");
            }
        }

        data.resize(width * height);
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                data[index(x, y)] = content[y][x];
            }
        }
    }
    Matrix(const Matrix& other)
        : width(other.width), height(other.height) {
        const size_t n = elementCount();
        if (n == 0) return;

        if constexpr (supports_gpu) {
            if (other.isDeviceValid()) {
                d_buf.ensure_capacity(n);
                cuda_backend::copy_d2d(d_buf.get(), other.d_buf.get(), n * sizeof(T));
                hostDirty = true;
                deviceDirty = false;
                return;
            }
        }

        data = other.data;
        hostDirty = false;
        deviceDirty = true;
    }
    Matrix& operator=(const Matrix& other) {
        if (this == &other) return *this;

        width = other.width;
        height = other.height;
        const size_t n = elementCount();

        if (n == 0) {
            data.clear();
            hostDirty = false;
            deviceDirty = true;
            return *this;
        }

        if constexpr (supports_gpu) {
            if (other.isDeviceValid()) {
                d_buf.ensure_capacity(n);
                cuda_backend::copy_d2d(d_buf.get(), other.d_buf.get(), n * sizeof(T));
                hostDirty = true;
                deviceDirty = false;
                return *this;
            }
        }

        data = other.data;
        hostDirty = false;
        deviceDirty = true;
        return *this;
    }

    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(Matrix&&) noexcept = default;
    ~Matrix() = default;

    // GPU interactions
    [[nodiscard]] bool isDeviceValid() const noexcept {
        return d_buf.allocated() && !deviceDirty;
    }
    [[nodiscard]] bool isHostDirty() const noexcept { return hostDirty; }
    [[nodiscard]] bool isDeviceDirty() const noexcept { return deviceDirty; }
    void syncToDevice() const {
        if constexpr (!supports_gpu) return;
        const size_t n = elementCount();
        if (n == 0 || isDeviceValid()) return;

        d_buf.ensure_capacity(n);
        if (data.size() != n) {
            data.resize(n, T{});
        }
        cuda_backend::copy_h2d(d_buf.get(), data.data(), n * sizeof(T));
        deviceDirty = false;
    }
    void syncToHost() const {
        if (!hostDirty) return;
        const size_t n = elementCount();
        data.resize(n);
        if constexpr (supports_gpu) {
            if (n > 0 && d_buf.allocated()) {
                cuda_backend::copy_d2h(data.data(), d_buf.get(), n * sizeof(T));
            }
        }
        hostDirty = false;
    }
    void toDevice() const {
        if (gpuAvailable()) syncToDevice();
    }
    void toHost() const {
        syncToHost();
    }
    void releaseDevice() {
        syncToHost();
        d_buf.release();
        deviceDirty = true;
    }
    // operations
    [[nodiscard]] Matrix<T, L> add_horizontal(const Matrix<T, L>& rowVec) const {
        if (rowVec.height != 1 || rowVec.width != width) {
            throw std::invalid_argument("Bias matrix must have dimensions 1 x matrix_width.");
        }

        if constexpr (supports_gpu) {
            if (gpuAvailable() && (isDeviceValid() || rowVec.isDeviceValid())) {
                syncToDevice();
                rowVec.syncToDevice();

                Matrix<T, L> result = createOnDevice(height, width);
                cuda_backend::add_horizontal_device<T>(
                    d_buf.get(), rowVec.d_buf.get(), result.d_buf.get(),
                    height, width, L == MatrixLayout::RowMajor, T{ 1 }
                );
                return result;
            }
        }

        syncToHost();
        rowVec.syncToHost();
        Matrix<T, L> result(*this);
        auto row_indices = std::views::iota(size_t(0), height);
        std::for_each(std::execution::par_unseq, row_indices.begin(), row_indices.end(), [&](size_t y) {
            for (size_t x = 0; x < width; ++x) {
                result.data[result.index(x, y)] += rowVec.data[x];
            }
            });
        return result;
    }

    [[nodiscard]] Matrix<T, L> add_horizontal(const std::vector<T>& horizontalVec) const {
        if (horizontalVec.size() != width)
            throw std::invalid_argument("Vector size must match matrix width.");

        if constexpr (supports_gpu) {
            if (isDeviceValid()) {
                cuda_backend::DeviceBuffer<T> d_vec;
                d_vec.ensure_capacity(width);
                cuda_backend::copy_h2d(d_vec.get(), horizontalVec.data(), width * sizeof(T));

                Matrix<T, L> result = createOnDevice(height, width);
                cuda_backend::add_horizontal_device<T>(
                    d_buf.get(), d_vec.get(), result.d_buf.get(),
                    height, width, L == MatrixLayout::RowMajor, T{ 1 }
                );
                return result;
            }
        }

        syncToHost();
        Matrix<T, L> result(*this);
        auto row_indices = std::views::iota(size_t(0), height);
        std::for_each(std::execution::par_unseq, row_indices.begin(), row_indices.end(), [&](size_t y) {
            for (size_t x = 0; x < width; ++x) {
                result.data[result.index(x, y)] += horizontalVec[x];
            }
            });
        return result;
    }

    [[nodiscard]] Matrix<T, L> subtract_horizontal(const Matrix<T, L>& rowVec) const {
        if (rowVec.height != 1 || rowVec.width != width) {
            throw std::invalid_argument("Row matrix must have dimensions 1 x matrix_width.");
        }

        if constexpr (supports_gpu) {
            if (gpuAvailable() && (isDeviceValid() || rowVec.isDeviceValid())) {
                syncToDevice();
                rowVec.syncToDevice();

                Matrix<T, L> result = createOnDevice(height, width);
                cuda_backend::add_horizontal_device<T>(
                    d_buf.get(), rowVec.d_buf.get(), result.d_buf.get(),
                    height, width, L == MatrixLayout::RowMajor, static_cast<T>(-1)
                );
                return result;
            }
        }

        syncToHost();
        rowVec.syncToHost();
        Matrix<T, L> result(*this);
        auto row_indices = std::views::iota(size_t(0), height);
        std::for_each(std::execution::par_unseq, row_indices.begin(), row_indices.end(), [&](size_t y) {
            for (size_t x = 0; x < width; ++x) {
                result.data[result.index(x, y)] -= rowVec.data[x];
            }
            });
        return result;
    }

    [[nodiscard]] Matrix<T, L> subtract_horizontal(const std::vector<T>& horizontalVec) const {
        if (horizontalVec.size() != width)
            throw std::invalid_argument("Vector size must match matrix width.");

        if constexpr (supports_gpu) {
            if (isDeviceValid()) {
                cuda_backend::DeviceBuffer<T> d_vec;
                d_vec.ensure_capacity(width);
                cuda_backend::copy_h2d(d_vec.get(), horizontalVec.data(), width * sizeof(T));

                Matrix<T, L> result = createOnDevice(height, width);
                cuda_backend::add_horizontal_device<T>(
                    d_buf.get(), d_vec.get(), result.d_buf.get(),
                    height, width, L == MatrixLayout::RowMajor, static_cast<T>(-1)
                );
                return result;
            }
        }

        syncToHost();
        Matrix<T, L> result(*this);
        auto row_indices = std::views::iota(size_t(0), height);
        std::for_each(std::execution::par_unseq, row_indices.begin(), row_indices.end(), [&](size_t y) {
            for (size_t x = 0; x < width; ++x) {
                result.data[result.index(x, y)] -= horizontalVec[x];
            }
            });
        return result;
    }

    [[nodiscard]] Matrix<T, L> operator-(const Matrix<T, L>& other) const {
        if (width != other.width || height != other.height) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction.");
        }

        if constexpr (supports_gpu) {
            if (gpuAvailable() && (isDeviceValid() || other.isDeviceValid())) {
                syncToDevice();
                other.syncToDevice();
                Matrix<T, L> result = createOnDevice(height, width);
                cuda_backend::sub_device<T>(d_buf.get(), other.d_buf.get(), result.d_buf.get(), elementCount());
                return result;
            }
        }

        syncToHost();
        other.syncToHost();
        Matrix<T, L> result(height, width);
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = data[i] - other.data[i];
        return result;
    }

    [[nodiscard]] Matrix<T, L> operator*(const T a) const {
        if constexpr (supports_gpu) {
            if (isDeviceValid()) {
                Matrix<T, L> result = createOnDevice(height, width);
                cuda_backend::scale_device<T>(d_buf.get(), a, result.d_buf.get(), elementCount());
                return result;
            }
        }

        syncToHost();
        Matrix<T, L> result(*this);
        std::for_each(std::execution::par_unseq, result.data.begin(), result.data.end(), [&](T& v) {
            v *= a;
            });
        return result;
    }

    void sub_scaled_inplace(const Matrix<T, L>& grad, T lr) {
        if (width != grad.width || height != grad.height) {
            throw std::invalid_argument("Matrix dimensions must match for in-place update.");
        }

        if constexpr (supports_gpu) {
            if (gpuAvailable() && (isDeviceValid() || grad.isDeviceValid())) {
                syncToDevice();
                grad.syncToDevice();
                cuda_backend::axpy_sub_inplace_device<T>(d_buf.get(), grad.d_buf.get(), lr, elementCount());
                hostDirty = true;
                deviceDirty = false;
                return;
            }
        }

        syncToHost();
        grad.syncToHost();
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] -= lr * grad.data[i];
        }
        hostDirty = false;
        deviceDirty = true;
    }

    template <MatrixLayout L2>
    [[nodiscard]] Matrix<T, L> operator*(const Matrix<T, L2>& m2) const {
        if (width != m2.height)
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");

        constexpr size_t GPU_OPS_THRESHOLD = 128ULL * 128ULL * 128ULL;
        const size_t total_ops = height * width * m2.width;

        if constexpr (supports_gpu) {
            if (gpuAvailable() && (isDeviceValid() || m2.isDeviceValid() || total_ops >= GPU_OPS_THRESHOLD)) {
                return multiply_gpu(m2);
            }
        }
        return multiply_cpu(m2);
    }

    template <MatrixLayout L2>
    [[nodiscard]] Matrix<T, L> multiply_gpu(const Matrix<T, L2>& m2) const {
        syncToDevice();
        m2.syncToDevice();

        Matrix<T, L> result = createOnDevice(height, m2.width);
        cuda_backend::multiply_matrices_device<T>(
            d_buf.get(), height, width, L == MatrixLayout::RowMajor,
            m2.d_buf.get(), m2.width, L2 == MatrixLayout::RowMajor,
            result.d_buf.get(), L == MatrixLayout::RowMajor
        );
        return result;
    }

    template <MatrixLayout L2>
    [[nodiscard]] Matrix<T, L> multiply_cpu(const Matrix<T, L2>& m2) const {
        syncToHost();
        m2.syncToHost();

        Matrix<T, L> result(height, m2.width);

        if constexpr (L == MatrixLayout::RowMajor && L2 == MatrixLayout::RowMajor) {
            auto row_indices = std::views::iota(size_t(0), height);
            std::for_each(std::execution::par_unseq, row_indices.begin(), row_indices.end(), [&](size_t y) {
                for (size_t k = 0; k < width; ++k) {
                    T r = data[index(k, y)];
                    for (size_t x = 0; x < m2.width; ++x) {
                        result.data[result.index(x, y)] += r * m2.data[m2.index(x, k)];
                    }
                }
                });
        }
        else if constexpr (L == MatrixLayout::RowMajor && L2 == MatrixLayout::ColumnMajor) {
            auto row_indices = std::views::iota(size_t(0), height);
            std::for_each(std::execution::par_unseq, row_indices.begin(), row_indices.end(), [&](size_t y) {
                for (size_t x = 0; x < m2.width; ++x) {
                    T sum = 0;
                    for (size_t k = 0; k < width; ++k) {
                        sum += data[index(k, y)] * m2.data[m2.index(x, k)];
                    }
                    result.data[result.index(x, y)] = sum;
                }
                });
        }
        else if constexpr (L == MatrixLayout::ColumnMajor && L2 == MatrixLayout::ColumnMajor) {
            auto col_indices = std::views::iota(size_t(0), m2.width);
            std::for_each(std::execution::par_unseq, col_indices.begin(), col_indices.end(), [&](size_t x) {
                for (size_t k = 0; k < width; ++k) {
                    T r = m2.data[m2.index(x, k)];
                    for (size_t y = 0; y < height; ++y) {
                        result.data[result.index(x, y)] += data[index(k, y)] * r;
                    }
                }
                });
        }
        else if constexpr (L == MatrixLayout::ColumnMajor && L2 == MatrixLayout::RowMajor) {
            auto col_indices = std::views::iota(size_t(0), m2.width);
            std::for_each(std::execution::par_unseq, col_indices.begin(), col_indices.end(), [&](size_t x) {
                for (size_t k = 0; k < width; ++k) {
                    T r = m2.data[m2.index(x, k)];
                    for (size_t y = 0; y < height; ++y) {
                        result.data[result.index(x, y)] += data[index(k, y)] * r;
                    }
                }
                });
        }

        return result;
    }

    template <typename Func>
    [[nodiscard]] Matrix<T, L> apply(Func&& func) const {
        syncToHost();
        Matrix<T, L> result(height, width);
        std::transform(std::execution::par_unseq, data.begin(), data.end(), result.data.begin(),
            [&](const T& val) {
                return static_cast<T>(func(val));
            });
        return result;
    }

    [[nodiscard]] Matrix<T, L> apply_activation(cuda_backend::ActivationOp op) const {
        if constexpr (supports_gpu) {
            if (isDeviceValid()) {
                Matrix<T, L> result = createOnDevice(height, width);
                cuda_backend::activation_device<T>(d_buf.get(), result.d_buf.get(), elementCount(), op);
                return result;
            }
        }

        return apply([op](T x) -> double {
            const double d = static_cast<double>(x);
            switch (op) {
            case cuda_backend::ActivationOp::ReLU:    return d > 0.0 ? d : 0.0;
            case cuda_backend::ActivationOp::Sigmoid: return 1.0 / (1.0 + std::exp(-d));
            case cuda_backend::ActivationOp::Tanh:    return std::tanh(d);
            case cuda_backend::ActivationOp::Linear:
            default:                                  return d;
            }
            });
    }

    [[nodiscard]] Matrix<T, L> apply_activation_derivative(cuda_backend::ActivationOp op) const {
        if constexpr (supports_gpu) {
            if (isDeviceValid()) {
                Matrix<T, L> result = createOnDevice(height, width);
                cuda_backend::activation_deriv_device<T>(d_buf.get(), result.d_buf.get(), elementCount(), op);
                return result;
            }
        }

        return apply([op](T a) -> double {
            const double d = static_cast<double>(a);
            switch (op) {
            case cuda_backend::ActivationOp::ReLU:    return d > 0.0 ? 1.0 : 0.0;
            case cuda_backend::ActivationOp::Sigmoid: return d * (1.0 - d);
            case cuda_backend::ActivationOp::Tanh:    return 1.0 - d * d;
            case cuda_backend::ActivationOp::Linear:
            default:                                  return 1.0;
            }
            });
    }

    [[nodiscard]] Matrix<T, L> softmax_rows() const {
        if constexpr (supports_gpu) {
            if (isDeviceValid()) {
                Matrix<T, L> result = createOnDevice(height, width);
                cuda_backend::softmax_rows_device<T>(
                    d_buf.get(), result.d_buf.get(), height, width, L == MatrixLayout::RowMajor
                );
                return result;
            }
        }

        syncToHost();
        Matrix<T, L> result(height, width);
        for (size_t y = 0; y < height; ++y) {
            if (width == 0) continue;
            double maxVal = static_cast<double>(data[index(0, y)]);
            for (size_t x = 1; x < width; ++x) {
                maxVal = std::max(maxVal, static_cast<double>(data[index(x, y)]));
            }
            double sumExp = 0.0;
            for (size_t x = 0; x < width; ++x) {
                const double e = std::exp(static_cast<double>(data[index(x, y)]) - maxVal);
                result.data[result.index(x, y)] = static_cast<T>(e);
                sumExp += e;
            }
            for (size_t x = 0; x < width; ++x) {
                result.data[result.index(x, y)] = static_cast<T>(
                    static_cast<double>(result.data[result.index(x, y)]) / sumExp
                    );
            }
        }
        return result;
    }

    [[nodiscard]] Matrix<T, L> hadamard(const Matrix<T, L>& other) const {
        if (width != other.width || height != other.height) {
            throw std::invalid_argument("Matrix dimensions must match for Hadamard product.");
        }

        if constexpr (supports_gpu) {
            if (gpuAvailable() && (isDeviceValid() || other.isDeviceValid())) {
                syncToDevice();
                other.syncToDevice();
                Matrix<T, L> result = createOnDevice(height, width);
                cuda_backend::hadamard_device<T>(d_buf.get(), other.d_buf.get(), result.d_buf.get(), elementCount());
                return result;
            }
        }

        syncToHost();
        other.syncToHost();
        Matrix<T, L> result(height, width);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix<T, L>& matrix) {
        os << matrix.toString();
        return os;
    }

    // simple functions
    [[nodiscard]] size_t elementCount() const noexcept {
        return width * height;
    }

    [[nodiscard]] static Matrix<T, L> identity(size_t n) {
        Matrix<T, L> m(n, n);
        constexpr T one = static_cast<T>(1);
        for (size_t i = 0; i < n; ++i) {
            m.setValue(i, i, one);
        }
        return m;
    }

    void fill(T val = T{0}) {
        if constexpr (supports_gpu) {
            if (isDeviceValid()) {
                cuda_backend::memset_device_fill(d_buf.get(), elementCount() * sizeof(T), val);
                hostDirty = true;
                deviceDirty = false;
                return;
            }
        }
        std::fill(data.begin(), data.end(), val);
        hostDirty = false;
        deviceDirty = true;
    }

    std::string toString(size_t precision = DEFAULT_PRINT_PRECISION) const {
        syncToHost();
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(precision);

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                oss << std::setw(10) << data[index(x, y)] << "\t";
            }
            oss << "\n";
        }
        return oss.str();
    }

    void print(size_t precision = DEFAULT_PRINT_PRECISION) const {
        std::cout << toString(precision);
    }

    void fillRandom(int a, int b) {
        static std::random_device rd;
        thread_local std::mt19937 gen(rd());
        data.resize(elementCount());

        if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dist(a, b);
            for (auto& val : data) val = dist(gen);
        }
        else {
            std::uniform_real_distribution<T> dist(a, b);
            for (auto& val : data) val = dist(gen);
        }
        hostDirty = false;
        deviceDirty = true;
    }

    [[nodiscard]] const std::vector<T>& getContent() const {
        syncToHost();
        return data;
    }

    void setContent(std::vector<T> newContent) {
        if (newContent.size() != width * height) throw std::invalid_argument("Size mismatch");
        data = std::move(newContent);
        hostDirty = false;
        deviceDirty = true;
    }
    [[nodiscard]] Dimensions getDimensions() const { return { width, height }; }
    [[nodiscard]] size_t getWidth() const { return width; }
    [[nodiscard]] size_t getHeight() const { return height; }
    [[nodiscard]] T getValue(size_t x, size_t y) const {
        syncToHost();
        return data[index(x, y)];
    }

    void setValue(size_t x, size_t y, T value) {
        syncToHost();
        data[index(x, y)] = value;
        deviceDirty = true;
    }

    [[nodiscard]] Matrix<T, L> mean_columns_matrix() const {
        if (height == 0 || width == 0)
            return Matrix<T, L>(1, width);
        
        if constexpr (supports_gpu) {
            if (isDeviceValid()) {
                Matrix<T, L> result = createOnDevice(1, width);
                cuda_backend::mean_columns_device<T>(
                    d_buf.get(), result.d_buf.get(), height, width, L == MatrixLayout::RowMajor
                );
                return result;
            }
        }

        syncToHost();
        Matrix<T, L> result(1, width);
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                result.data[x] += data[index(x, y)];
            }
        }
        const T invHeight = static_cast<T>(1.0) / static_cast<T>(height);
        for (size_t x = 0; x < width; ++x) {
            result.data[x] *= invHeight;
        }
        return result;
    }

    [[nodiscard]] std::vector<double> getMeanColumns() const {
        if (height == 0 || width == 0) {
            return std::vector<double>(width, 0.0);
        }
        Matrix<T, L> m = mean_columns_matrix();
        const auto& content = m.getContent();
        std::vector<double> res(content.size());
        for (size_t i = 0; i < content.size(); ++i) {
            res[i] = static_cast<double>(content[i]);
        }
        return res;
    }

    [[nodiscard]] double getGrandMean() const {
        syncToHost();
        if (data.empty()) return 0.0;
        double sum = 0.0;
        for (const auto& val : data) {
            sum += static_cast<double>(val);
        }
        return sum / static_cast<double>(data.size());
    }

    [[nodiscard]] Matrix<T, L> transposed() const {
        if constexpr (supports_gpu) {
            if (isDeviceValid()) {
                Matrix<T, L> transposedM = createOnDevice(width, height);
                cuda_backend::transpose_device<T>(
                    d_buf.get(), transposedM.d_buf.get(), height, width, L == MatrixLayout::RowMajor
                );
                return transposedM;
            }
        }

        syncToHost();
        Matrix<T, L> transposedM(width, height); // rows = width, cols = height
        auto row_indices = std::views::iota(size_t(0), height);

        std::for_each(std::execution::par_unseq, row_indices.begin(), row_indices.end(), [&](size_t y) {
            for (size_t x = 0; x < width; ++x) {
                transposedM.data[transposedM.index(y, x)] = data[index(x, y)];
            }
            });
        return transposedM;
    }
};