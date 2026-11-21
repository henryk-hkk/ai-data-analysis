#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "Concepts.h"
#include "MatrixFNN.h"
#include "Dataset.h"
#include "mfuncs.h"

class MatrixFNNTrainingAgent {
public:
    template<typename T, MatrixLayout L, typename DatasetT>
        requires arithmetic<T>
    static void trainFNN(
        MatrixFNN<T, L>& fnn,
        const DatasetT& trainingDataset,
        unsigned epochs,
        std::type_identity_t<T> learningRate,
        size_t batchSize = 32,
        bool shuffle = true,
        bool getConsoleOutput = false,
        bool useGpu = cuda_backend::is_gpu_available()
    ) {
        if (fnn.layers.empty()) {
            throw std::logic_error("Cannot train a network with no weighted layers.");
        }
        if (batchSize == 0) {
            throw std::invalid_argument("Batch size must be greater than 0.");
        }

        const size_t datasetSize = trainingDataset.inputVector.size();
        if (datasetSize == 0) {
            return;
        }

        if (useGpu) {
            fnn.toDevice();
        }

        std::vector<size_t> indices(datasetSize);
        std::iota(indices.begin(), indices.end(), 0);
        thread_local std::mt19937 gen(std::random_device{}());

        for (unsigned epoch = 0; epoch < epochs; ++epoch) {
            if (shuffle) {
                std::shuffle(indices.begin(), indices.end(), gen);
            }

            double epochCostSum = 0.0;
            size_t batchCount = 0;

            for (size_t start = 0; start < datasetSize; start += batchSize) {
                const size_t currentBatchSize = std::min(batchSize, datasetSize - start);

                Matrix<T, L> inputBatch = buildBatchMatrix<T, L>(
                    trainingDataset.inputVector, indices, start, currentBatchSize
                );
                Matrix<T, L> targetBatch = buildBatchMatrix<T, L>(
                    trainingDataset.desiredOutputVector, indices, start, currentBatchSize
                );

                if (useGpu) {
                    inputBatch.toDevice();
                    targetBatch.toDevice();
                }

                std::vector<Matrix<T, L>> activations = forwardWithCache(fnn, inputBatch);

                if (getConsoleOutput) {
                    epochCostSum += getBatchCost(activations.back(), targetBatch);
                    ++batchCount;
                }

                backwardAndUpdate(fnn, activations, targetBatch, learningRate);
            }

            if (getConsoleOutput && batchCount > 0) {
                std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                    << " | Mean batch cost: " << (epochCostSum / static_cast<double>(batchCount))
                    << "\n";
            }
        }
    }

    template<typename T, MatrixLayout L, typename DatasetT>
        requires arithmetic<T>
    [[nodiscard]] static double testFNN(
        const MatrixFNN<T, L>& fnn,
        const DatasetT& testDataset,
        size_t batchSize = 64,
        bool useGpu = cuda_backend::is_gpu_available()
    ) {
        if (fnn.layers.empty()) {
            throw std::logic_error("Cannot test a network with no weighted layers.");
        }

        const size_t datasetSize = testDataset.inputVector.size();
        if (datasetSize == 0) {
            return 0.0;
        }

        if (useGpu) {
            fnn.toDevice();
        }

        std::vector<size_t> indices(datasetSize);
        std::iota(indices.begin(), indices.end(), 0);

        size_t correct = 0;
        const size_t outWidth = fnn.layers.back().biases.getWidth();

        for (size_t start = 0; start < datasetSize; start += batchSize) {
            const size_t currentBatchSize = std::min(batchSize, datasetSize - start);
            Matrix<T, L> inputBatch = buildBatchMatrix<T, L>(
                testDataset.inputVector, indices, start, currentBatchSize
            );

            if (useGpu) {
                inputBatch.toDevice();
            }

            Matrix<T, L> outputBatch = fnn.forward(inputBatch);
            for (size_t row = 0; row < currentBatchSize; ++row) {
                std::vector<double> predicted(outWidth);
                for (size_t col = 0; col < outWidth; ++col) {
                    predicted[col] = static_cast<double>(outputBatch.getValue(col, row));
                }

                const auto& expected = testDataset.desiredOutputVector[start + row];
                if (mfuncs::convertOutputVectorToDigit(predicted) ==
                    mfuncs::convertOutputVectorToDigit(expected)) {
                    ++correct;
                }
            }
        }

        return static_cast<double>(correct) / static_cast<double>(datasetSize);
    }

protected:
    template<typename T, MatrixLayout L, typename SourceVec>
        requires arithmetic<T>
    static Matrix<T, L> buildBatchMatrix(
        const SourceVec& source,
        const std::vector<size_t>& indices,
        size_t start,
        size_t count
    ) {
        const size_t cols = source[indices[start]].size();
        Matrix<T, L> batch(count, cols); // rows = count, cols = cols

        for (size_t row = 0; row < count; ++row) {
            const auto& sample = source[indices[start + row]];
            for (size_t col = 0; col < cols; ++col) {
                batch.setValue(col, row, static_cast<T>(sample[col]));
            }
        }
        return batch;
    }

    template<typename T, MatrixLayout L>
        requires arithmetic<T>
    static std::vector<Matrix<T, L>> forwardWithCache(
        const MatrixFNN<T, L>& fnn,
        const Matrix<T, L>& inputBatch
    ) {
        std::vector<Matrix<T, L>> activations;
        activations.reserve(fnn.layers.size() + 1);
        activations.push_back(inputBatch);

        for (const auto& layer : fnn.layers) {
            Matrix<T, L> current = (activations.back() * layer.weights.transposed())
                .add_horizontal(layer.biases);

            if (layer.func == ActivationFunction::SOFTMAX) {
                current = current.softmax_rows();
            }
            else if (layer.func != ActivationFunction::LINEAR) {
                if (auto gpuOp = toCudaActivation(layer.func)) {
                    current = current.apply_activation(*gpuOp);
                }
                else {
                    current = current.apply(getActivationFunction(layer.func));
                }
            }

            activations.push_back(std::move(current));
        }

        return activations;
    }

    template<typename T, MatrixLayout L>
        requires arithmetic<T>
    static void backwardAndUpdate(
        MatrixFNN<T, L>& fnn,
        const std::vector<Matrix<T, L>>& activations,
        const Matrix<T, L>& targetBatch,
        T learningRate
    ) {
        const size_t numLayers = fnn.layers.size();
        const size_t batchSize = targetBatch.getHeight();
        const T invBatch = static_cast<T>(1.0) / static_cast<T>(batchSize);

        // softmax + cross-entropy -> (A - Y)
        // other -> 2 * (A - Y) * f'(A)
        Matrix<T, L> delta = activations.back() - targetBatch;
        const ActivationFunction outFunc = fnn.layers.back().func;

        if (outFunc != ActivationFunction::SOFTMAX) {
            if (outFunc != ActivationFunction::LINEAR) {
                if (auto gpuOp = toCudaActivation(outFunc)) {
                    delta = delta.hadamard(activations.back().apply_activation_derivative(*gpuOp));
                }
                else {
                    delta = delta.hadamard(activations.back().apply(getDerivative(outFunc)));
                }
            }
            delta = delta * static_cast<T>(2);
        }

        for (size_t l = numLayers; l-- > 0;) {
            auto& layer = fnn.layers[l];
            const Matrix<T, L>& prevAct = activations[l];

            Matrix<T, L> dW = (delta.transposed() * prevAct) * invBatch;
            Matrix<T, L> dB = delta.mean_columns_matrix();

            Matrix<T, L> nextDelta;
            if (l > 0) {
                nextDelta = delta * layer.weights;
                const ActivationFunction prevFunc = fnn.layers[l - 1].func;
                if (prevFunc != ActivationFunction::LINEAR) {
                    if (auto gpuOp = toCudaActivation(prevFunc)) {
                        nextDelta = nextDelta.hadamard(prevAct.apply_activation_derivative(*gpuOp));
                    }
                    else {
                        nextDelta = nextDelta.hadamard(prevAct.apply(getDerivative(prevFunc)));
                    }
                }
            }

            layer.weights.sub_scaled_inplace(dW, learningRate);
            layer.biases.sub_scaled_inplace(dB, learningRate);

            if (l > 0) {
                delta = std::move(nextDelta);
            }
        }
    }

    template<typename T, MatrixLayout L>
        requires arithmetic<T>
    static double getBatchCost(
        const Matrix<T, L>& outputBatch,
        const Matrix<T, L>& targetBatch
    ) {
        Matrix<T, L> diff = outputBatch - targetBatch;
        double sumSq = 0.0;
        for (const T& val : diff.getContent()) {
            const double d = static_cast<double>(val);
            sumSq += d * d;
        }
        return sumSq / static_cast<double>(outputBatch.getHeight());
    }
};