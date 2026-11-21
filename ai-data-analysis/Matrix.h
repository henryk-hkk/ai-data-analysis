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
#include "Concepts.h"
#include "MatrixLayout.h"
#include "Dimensions.h"
#include "CudaBackend.hpp"

template<typename T = double, MatrixLayout L = MatrixLayout::RowMajor>
    requires arithmetic<T>
class Matrix {
private:
    size_t width, height;
    std::vector<T> data;
    static constexpr size_t DEFAULT_PRINT_PRECISION = 6;

    template<typename U, MatrixLayout L2>
        requires arithmetic<U>
    friend class Matrix;

    constexpr size_t index(size_t x, size_t y) const {
        if constexpr (L == MatrixLayout::RowMajor)
            return y * width + x;
        else return x * height + y;
    }

public:
    Matrix() : width(0), height(0) {}

    Matrix(size_t width, size_t height)
        : width(width), height(height), data(width* height, T{}) {
    }

    Matrix(size_t width, size_t height, std::vector<T> content)
        : width(width), height(height), data(std::move(content)) {
        if (data.size() != width * height)
            throw std::invalid_argument("Vector size does not match matrix dimensions.");
    }

    [[nodiscard]] Matrix<T, L> add_horizontal(const std::vector<T>& horizontalVec) const {
        if (horizontalVec.size() != width)
            throw std::invalid_argument("Vector size must match matrix width.");

        Matrix<T, L> result(*this);
        auto row_indices = std::views::iota(size_t(0), height);

        std::for_each(std::execution::par_unseq, row_indices.begin(), row_indices.end(), [&](size_t y) {
            for (size_t x = 0; x < width; ++x) {
                result.data[result.index(x, y)] += horizontalVec[x];
            }
            });
        return result;
    }

    [[nodiscard]] Matrix<T, L> subtract_horizontal(const std::vector<T>& horizontalVec) const {
        if (horizontalVec.size() != width) {
            throw std::invalid_argument("Vector size must match matrix width.");
        }

        Matrix<T, L> result(*this);
        auto row_indices = std::views::iota(size_t(0), height);

        std::for_each(std::execution::par_unseq, row_indices.begin(), row_indices.end(), [&](size_t y) {
            for (size_t x = 0; x < width; ++x) {
                result.data[result.index(x, y)] -= horizontalVec[x];
            }
            });
        return result;
    }

    [[nodiscard]] Matrix<T, L> operator*(const T a) const {
        Matrix<T, L> result(*this);
        auto all_indices = std::views::iota(size_t(0), result.data.size());

        std::for_each(std::execution::par_unseq, all_indices.begin(), all_indices.end(), [&](size_t i) {
            result.data[i] *= a;
            });
        return result;
    }
    template <MatrixLayout L2>
    [[nodiscard]] Matrix<T, L> operator*(const Matrix<T, L2>& m2) const {
        if (width != m2.height)
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");

        constexpr size_t GPU_OPS_THRESHOLD = 128ULL * 128ULL * 128ULL;
        const size_t total_ops = height * width * m2.width;

        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, int>) {
            static const bool has_gpu = cuda_backend::is_gpu_available();
            if (has_gpu && total_ops >= GPU_OPS_THRESHOLD) {
                return multiply_gpu(m2);
            }
        }
        return multiply_cpu(m2);
    }

    template <MatrixLayout L2>
    [[nodiscard]] Matrix<T, L> multiply_gpu(const Matrix<T, L2>& m2) const {
        Matrix<T, L> result(m2.width, height);

        cuda_backend::multiply_matrices<T>(
            data.data(), height, width, L == MatrixLayout::RowMajor,
            m2.data.data(), m2.width, L2 == MatrixLayout::RowMajor,
            result.data.data(), L == MatrixLayout::RowMajor
        );

        return result;
    }
    template <MatrixLayout L2>
    [[nodiscard]] Matrix<T, L> multiply_cpu(const Matrix<T, L2>& m2) const {
        Matrix<T, L> result(m2.width, height);

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
                for (size_t y = 0; y < height; ++y) {
                    T sum = 0;
                    for (size_t k = 0; k < width; ++k) {
                        sum += data[index(k, y)] * m2.data[m2.index(x, k)];
                    }
                    result.data[result.index(x, y)] = sum;
                }
                });
        }

        return result;
    }

    template <typename Func>
    [[nodiscard]] Matrix<T, L> apply(Func&& func) const {
        Matrix<T, L> result(width, height);
        auto all_indices = std::views::iota(size_t(0), data.size());

        std::for_each(std::execution::par_unseq, all_indices.begin(), all_indices.end(), [&](size_t i) {
            result.data[i] = static_cast<T>(func(data[i]));
            });
        return result;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix<T, L>& matrix) {
        os << matrix.toString();
        return os;
    }

    [[nodiscard]] static Matrix<T, L> identity(size_t n) {
        Matrix<T, L> m(n, n);
        constexpr T one = static_cast<T>(1);
        auto indices = std::views::iota(size_t(0), n);

        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](size_t i) {
            m.setValue(i, i, one);
            });
        return m;
    }

    void clear() {
        auto all_indices = std::views::iota(size_t(0), data.size());
        std::for_each(std::execution::par_unseq, all_indices.begin(), all_indices.end(), [&](size_t i) {
            data[i] = T{ 0 };
            });
    }

    std::string toString(size_t precision = DEFAULT_PRINT_PRECISION) const {
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
        std::cout << toString();
    }

    void fillWithRandomIntegers(int a, int b) {
        static std::random_device rd;
        static std::mt19937 gen(rd());

        if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dist(a, b);
            for (auto& val : data) val = dist(gen);
        }
        else {
            std::uniform_real_distribution<T> dist(a, b);
            for (auto& val : data) val = dist(gen);
        }
    }

    [[nodiscard]] const std::vector<T>& getContent() const { return data; }
    void setContent(std::vector<T> newContent) {
        if (newContent.size() != width * height) throw std::invalid_argument("Size mismatch");
        data = std::move(newContent);
    }

    [[nodiscard]] Dimensions getDimensions() const { return { width, height }; }
    [[nodiscard]] size_t getWidth() const { return width; }
    [[nodiscard]] size_t getHeight() const { return height; }

    [[nodiscard]] T getValue(size_t x, size_t y) const {
        return data[index(x, y)];
    }

    void setValue(size_t x, size_t y, T value) {
        data[index(x, y)] = value;
    }

    [[nodiscard]] std::vector<double> getMeanColumns() const {
        std::vector<double> columnsMean(width, 0.0);
        if (height == 0) return columnsMean;

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                columnsMean[x] += static_cast<double>(data[index(x, y)]);
            }
        }
        for (size_t x = 0; x < width; ++x) {
            columnsMean[x] /= height;
        }
        return columnsMean;
    }

    [[nodiscard]] double getGrandMean() const {
        if (data.empty()) return 0.0;
        double sum = 0.0;
        for (const auto& val : data) {
            sum += static_cast<double>(val);
        }
        return sum / data.size();
    }

    [[nodiscard]] Matrix<T, L> transposed() const {
        Matrix<T, L> transposedM(height, width);
        auto row_indices = std::views::iota(size_t(0), height);

        std::for_each(std::execution::par_unseq, row_indices.begin(), row_indices.end(), [&](size_t y) {
            for (size_t x = 0; x < width; ++x) {
                transposedM.setValue(y, x, this->getValue(x, y));
            }
            });
        return transposedM;
    }
};