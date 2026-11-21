#pragma once
#include "Concepts.h"
#include "MatrixLayout.h"
#include "ActivationFunction.h"

template<typename T = double>
    requires arithmetic<T>
struct FNNInputLayerMatrixForm {
    ActivationFunction func;
    std::vector<T> biases;
};

template<typename T = double, MatrixLayout L = MatrixLayout::RowMajor>
    requires arithmetic<T>
struct FNNWeightedLayerMatrixForm {
    ActivationFunction func;
    Matrix<T, L> biases;  // 1 x N_out
    Matrix<T, L> weights; // N_out x N_in
};