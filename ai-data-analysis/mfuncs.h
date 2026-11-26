#pragma once
#include <vector>
#include <cmath>
#include <cstdlib>
#include "Concepts.h"
#include "Matrix.hpp"

template<std::floating_point T>
constexpr T RELU_LEAKAGE_RATE = T{ 0.05 };

namespace mfuncs { //Mathematical functions
	//get max value
    template<arithmetic T>
    [[nodiscard]] constexpr T max(T a, T b) noexcept {
        return (a > b) ? a : b;
    }

    template<std::ranges::input_range R>
        requires arithmetic<std::ranges::range_value_t<R>>
    [[nodiscard]] constexpr auto max(const R& range) {
        return std::ranges::max(range);
    }

    // Activation functions and their derivatives
    [[nodiscard]] constexpr auto linear(std::floating_point auto x) noexcept {
        return x;
    }

    template<std::floating_point T>
    [[nodiscard]] T sigmoid(T x) noexcept {
        return T{ 1 } / (T{ 1 } + std::exp(-x));
    }
    template<std::floating_point T>
    [[nodiscard]] constexpr T dSigmoid(T x) noexcept { // x - output of sigmoid
        return x * (T{ 1 } - x);
    }

    template<std::floating_point T>
    [[nodiscard]] constexpr T ReLU(T x) noexcept {
        return max(T{ 0 }, x);
    }
    template<std::floating_point T>
    [[nodiscard]] constexpr T dReLU(T x) noexcept {
        return (x >= T{ 0 }) ? T{ 1 } : T{ 0 };
    }

    template<std::floating_point T>
    [[nodiscard]] constexpr T lReLU(T x) noexcept { // Leaky ReLU
        return max(RELU_LEAKAGE_RATE<T> * x, x);
    }
    template<std::floating_point T>
    [[nodiscard]] constexpr T dLReLU(T x) noexcept { // Leaky ReLU derivative
        return (x >= RELU_LEAKAGE_RATE<T> ? T{ 1 } : RELU_LEAKAGE_RATE<T>);
    }

    template<std::floating_point T>
    [[nodiscard]] constexpr T tanh(T x) { // hyperbolic tangent function
        return static_cast<T>((std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x)));
    }
    template<std::floating_point T>
    [[nodiscard]] constexpr T dTanh(T x) { // tanh function derivative (x - output of a tanh function)
        return T{ 1 } - (x * x);
    }
	std::vector<double> softmax(const std::vector<double>& vec);
	//random number generation
	double getRandomDouble(double a, double b);
	int getRandomInteger(int a, int b);

	//vector-digit conversion
    template<arithmetic T>
    std::vector<T> convertDigitToOutputVector(unsigned digit) {
        std::vector<T> outputVector(10, 0);
        outputVector[digit] = 1;
        return outputVector;
    }
    template<arithmetic T>
    unsigned convertOutputVectorToDigit(const std::vector<T>& outputVector) {
        unsigned digit = 0;
        T max = outputVector[0];
        for (size_t i{ 1 }; i < outputVector.size(); i++) {
            if (outputVector[i] > max) {
                max = outputVector[i];
                digit = i;
            }
        }
        return digit;
    }
	//vector operations
	std::vector<double> combineVectors(const std::vector<double>& vec1, const std::vector<double>& vec2);
	std::vector<double> addVectors(const std::vector<double>& vec1, const std::vector<double>& vec2);
	std::vector<double> processVector(const std::vector<double>& input, double (*function)(double x));
	std::vector<double> hadamard(const std::vector<double>& vec1, const std::vector<double>& vec2);

    template <typename T>
        requires arithmetic<T>
    [[nodiscard]] T signum(T n) {
        return (n >= T{ 0 }) ? T{ 1 } : T{ -1 };
    }

    template <arithmetic T>
    [[nodiscard]] T norm(const std::vector<T>& vec) {
        T sqSum{ 0 };
        for (const auto& v : vec) {
            sqSum += v * v;
        }
        T norm = std::sqrt(sqSum);

        if (norm <= std::numeric_limits<T>::epsilon()) {
            return T{ 0 };
        }
        return norm;
    }

    template <typename T>
        requires arithmetic<T>
    void normalise(std::vector<T>& vec) {
        if (vec.empty()) return;

        T vecNorm = norm(vec);

        if (vecNorm <= std::numeric_limits<T>::epsilon()) {
            vec[0] = T{ 1 };
            for (size_t i = 1; i < vec.size(); ++i) {
                vec[i] = T{ 0 };
            }
            return;
        }

        for (auto& v : vec) {
            v /= vecNorm;
        }
    }

    template<typename T, MatrixLayout L>
        requires arithmetic<T>
    [[nodiscard]] Matrix<T, L> getSqrtValues(const Matrix<T, L>& m) {
        std::vector<T> content = m.getContent();

        for (auto& val : content) {
            val = std::sqrt(std::abs(val));
        }

        return Matrix<T, L>(m.getWidth(), m.getHeight(), std::move(content));
    }

    template<typename T, MatrixLayout L>
        requires arithmetic<T>
    [[nodiscard]] Matrix<T, L> getMeanCenteredMatrix(const Matrix<T, L>& m) {
        return m.subtract_horizontal(m.getMeanColumns());
    }
};