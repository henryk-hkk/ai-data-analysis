#pragma once
#include <vector>
#include <cmath>
#include <cstdlib>
#include "Concepts.h"
#include "Matrix.h"

namespace mfuncs { //Mathematical Functions
	//get max value
	double max(double a, double b);
	double max(const std::vector<double>& vec);
	//activation functions and their derivatives
	double linear(double x);
	double sigmoid(double x);
	double dSigmoid(double x);
	double ReLU(double x);
	double dReLU(double x);
	double tanh(double x);
	double dTanh(double x);
	std::vector<double> softmax(const std::vector<double>& vec);
	//random number generation
	double getRandomDouble(double a, double b);
	int getRandomInteger(int a, int b);
	//vector-digit conversion
	std::vector<double> convertDigitToOutputVector(unsigned digit);
	unsigned convertOutputVectorToDigit(const std::vector<double>& outputVector);
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

    template <typename T>
        requires arithmetic<T>
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