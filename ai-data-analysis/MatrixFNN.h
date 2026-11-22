#pragma once

#include <cmath>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include "Concepts.h"
#include "CudaBackend.hpp"
#include "FNNLayersMatrixForms.h"
#include "mfuncs.h"
#include "FNN.hpp"

inline std::optional<cuda_backend::ActivationOp> toCudaActivation(AFunc func) {
    switch (func) {
    case AFunc::LINEAR:  return cuda_backend::ActivationOp::Linear;
    case AFunc::RELU:    return cuda_backend::ActivationOp::ReLU;
    case AFunc::SIGMOID: return cuda_backend::ActivationOp::Sigmoid;
    case AFunc::TANH:    return cuda_backend::ActivationOp::Tanh;
    default:                          return std::nullopt;
    }
}

template<typename T = double, MatrixLayout L = MatrixLayout::RowMajor>
    requires arithmetic<T>
class MatrixFNN : public FNN<T> { // Linear-based Feedforward Neural Network
private:
    static std::vector<T> generateWeights(size_t fanIn, size_t fanOut, WeightInit init) {
        thread_local std::mt19937 gen(std::random_device{}());
        std::vector<T> values(fanIn * fanOut);

        if (init == WeightInit::Uniform) {
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (auto& v : values) {
                v = static_cast<T>(dist(gen));
            }
            return values;
        }

        // Xavier: stddev = sqrt(2 / (fanIn + fanOut))
        // He:     stddev = sqrt(2 / fanIn)
        const double stddev = (init == WeightInit::He)
            ? std::sqrt(2.0 / static_cast<double>(fanIn))
            : std::sqrt(2.0 / static_cast<double>(fanIn + fanOut));

        std::normal_distribution<double> dist(0.0, stddev);
        for (auto& v : values) {
            v = static_cast<T>(dist(gen));
        }
        return values;
    }

public:
    FNNInputLayerMatrixForm<T> inputLayer;
    std::vector<FNNWeightedLayerMatrixForm<T, L>> layers; // hidden + output

    // CUDA helpers
    void toDevice() const {
        for (const auto& layer : layers) {
            layer.weights.toDevice();
            layer.biases.toDevice();
        }
    }

    void toHost() const {
        for (const auto& layer : layers) {
            layer.weights.toHost();
            layer.biases.toHost();
        }
    }

    void setActivationFunction(AFunc func) {
        for (size_t i = 0; i + 1 < layers.size(); ++i) {
            layers[i].activationFunc = func;
        }
    }

    void setLayerActivationFunction(size_t layerIndex, AFunc func) {
        layers.at(layerIndex).activationFunc = func;
    }

    void setOutputLayerActivationFunction(AFunc func) {
        if (layers.empty()) {
            throw std::logic_error("Cannot set output activation: network has no weighted layers.");
        }
        layers.back().activationFunc = func;
    }

    void addInputLayer(size_t layerSize, AFunc func = AFunc::LINEAR) {
        if (layerSize == 0) {
            throw std::invalid_argument("Input layer size must be greater than 0.");
        }
        layers.clear();
        inputLayer.activationFunc = func;
        inputLayer.biases.assign(layerSize, T{ 0 });
    }

    void addLayer(
        size_t layerSize,
        AFunc func = AFunc::LINEAR,
        WeightInit init = WeightInit::Xavier
    ) {
        if (inputLayer.biases.empty()) {
            throw std::logic_error("Add an input layer before adding weighted layers.");
        }
        if (layerSize == 0) {
            throw std::invalid_argument("Layer size must be greater than 0.");
        }

        const size_t prevSize = layers.empty()
            ? inputLayer.biases.size()
            : layers.back().biases.getWidth();

        Matrix<T, L> weights(layerSize, prevSize, generateWeights(prevSize, layerSize, init));
        Matrix<T, L> biases(1, layerSize);

        layers.push_back(FNNWeightedLayerMatrixForm<T, L>{
            func,
                std::move(biases),
                std::move(weights)
        });
    }

    void addOutputLayer(
        size_t layerSize,
        AFunc func = AFunc::SOFTMAX,
        WeightInit init = WeightInit::Xavier
    ) {
        addLayer(layerSize, func, init);
    }

    [[nodiscard]] std::vector<T> forward(const std::vector<T>& input) {
        if (input.size() != inputLayer.biases.size()) {
            throw std::invalid_argument("Input size does not match input layer neuron count.");
        }

        Matrix<T, L> inputMatrix(1, input.size(), input);
        return forward(inputMatrix).getContent();
    }

    [[nodiscard]] Matrix<T, L> forward(const Matrix<T, L>& input) const {
        if (input.getWidth() != inputLayer.biases.size()) {
            throw std::invalid_argument("Input width does not match input layer neuron count.");
        }

        Matrix<T, L> current = input;

        for (const auto& layer : layers) {
            // current: (B x N_in), weights^T: (N_in x N_out) -> wynik: (B x N_out)
            current = (current * layer.weights.transposed()).add_horizontal(layer.biases);

            if (layer.activationFunc == AFunc::SOFTMAX) {
                current = current.softmax_rows();
            }
            else if (layer.activationFunc != AFunc::LINEAR) {
                if (auto gpuOp = toCudaActivation(layer.activationFunc)) {
                    current = current.apply_activation(*gpuOp);
                }
                else {
                    current = current.apply(getActivationFunction(layer.activationFunc));
                }
            }
        }

        return current;
    }
};