#pragma once
#include "Concepts.h"
#include "MatrixLayout.h"
#include "AFunc.h"

template<typename T = double>
    requires arithmetic<T>
struct FNNInputLayerMatrixForm {
    AFunc activationFunc;
    std::vector<T> biases;
};

template<typename T = double, MatrixLayout L = MatrixLayout::RowMajor>
    requires arithmetic<T>
struct FNNWeightedLayerMatrixForm {
    AFunc activationFunc;
    Matrix<T, L> biases;  // 1 x N_out
    Matrix<T, L> weights; // N_out x N_in
};