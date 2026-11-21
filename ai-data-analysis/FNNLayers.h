#pragma once

#include <cstdlib>
#include <ctime>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include "Concepts.h"
#include "Neuron.h"
#include "mfuncs.h"
#include "ActivationFunction.h"
#include "FNNLayersMatrixForms.h"
#include "LayerVectors.h"

template<typename T = double>
    requires arithmetic<T>
class FNNLayer {
public:
    FNNLayer* previous = nullptr;
    FNNLayer* next = nullptr;
    size_t neuronCount = 0;
    std::vector<std::vector<T>> weights; // weights[i][j] -> i: neuron bieżącej warstwy, j: neuron poprzedniej
    std::vector<std::unique_ptr<Neuron<T>>> neurons;
    std::vector<T> activations;
    ActivationFunction activationFunc = ActivationFunction::LINEAR;

    FNNLayer() = default;

    FNNLayer(size_t neuronCount, ActivationFunction func = ActivationFunction::LINEAR)
        : neuronCount(neuronCount), activations(neuronCount), activationFunc(func) {
    }

    FNNLayer(size_t neuronCount, FNNLayer* prev, ActivationFunction func = ActivationFunction::LINEAR)
        : previous(prev), neuronCount(neuronCount), activations(neuronCount), activationFunc(func) {
        init();
    }

    void init() {
        if (!previous) {
            throw std::logic_error("Cannot initialize weighted layer without a previous layer.");
        }

        static const bool seeded = []() {
            std::srand(static_cast<unsigned>(std::time(nullptr)));
            return true;
            }();
        (void)seeded;

        neurons.clear();
        neurons.reserve(neuronCount);
        for (size_t i = 0; i < neuronCount; ++i) {
            neurons.push_back(std::make_unique<Neuron<T>>(static_cast<T>(mfuncs::getRandomDouble(0, 1))));
        }

        weights.assign(neuronCount, std::vector<T>(previous->neuronCount));
        for (size_t i = 0; i < neuronCount; ++i) {
            for (size_t j = 0; j < previous->neuronCount; ++j) {
                weights[i][j] = static_cast<T>(mfuncs::getRandomInteger(-10, 10) * 0.1); // [-1; 1]
            }
        }
    }

    void processInput() {
        const std::vector<T>& prevActivations = previous->activations;
        auto actFunc = getActivationFunction(activationFunc);

        for (size_t i = 0; i < neuronCount; ++i) {
            T inputSum{};
            for (size_t j = 0; j < previous->neuronCount; ++j) {
                inputSum += prevActivations[j] * weights[i][j];
            }
            const T activated = static_cast<T>(actFunc(static_cast<double>(inputSum + neurons[i]->bias)));
            neurons[i]->activate(activated);
            activations[i] = neurons[i]->value;
        }
    }

    template<MatrixLayout L = MatrixLayout::RowMajor>
    [[nodiscard]] FNNWeightedLayerMatrixForm<T, L> toMatrixForm() const {
        LayerVectors<T> lV = getLayerVectors();
        const size_t nOut = lV.biases.size();
        return {
            activationFunc,
            Matrix<T, L>(1, nOut, std::move(lV.biases)),
            Matrix<T, L>(lV.weights)
        };
    }

    [[nodiscard]] LayerVectors<T> getLayerVectors() const {
        LayerVectors<T> lV;
        lV.weights = weights;
        lV.biases.reserve(neuronCount);
        for (size_t neuron = 0; neuron < neuronCount; ++neuron) {
            lV.biases.push_back(neurons[neuron]->bias);
        }
        return lV;
    }

    virtual ~FNNLayer() = default;
};

template<typename T = double>
    requires arithmetic<T>
struct FNNInputLayer : FNNLayer<T> {
    FNNInputLayer(size_t neuronCount, ActivationFunction func = ActivationFunction::LINEAR)
        : FNNLayer<T>(neuronCount, func) {
        this->neurons.reserve(neuronCount);
        for (size_t i = 0; i < neuronCount; ++i) {
            this->neurons.push_back(std::make_unique<Neuron<T>>());
        }
    }

    void processInput(const std::vector<T>& inputArr) {
        if (inputArr.size() != this->neuronCount) {
            throw std::invalid_argument("Input vector size does not match input layer neuron count.");
        }
        for (size_t i = 0; i < this->neuronCount; ++i) {
            this->neurons[i]->activate(inputArr[i]);
            this->activations[i] = this->neurons[i]->value;
        }
    }

    [[nodiscard]] FNNInputLayerMatrixForm<T> toMatrixForm() const {
        LayerVectors<T> lV = this->getLayerVectors();
        return { this->activationFunc, std::move(lV.biases) };
    }
};

template<typename T = double>
    requires arithmetic<T>
struct FNNHiddenLayer : FNNLayer<T> {
    using FNNLayer<T>::FNNLayer;
};

template<typename T = double>
    requires arithmetic<T>
struct FNNOutputLayer : FNNLayer<T> {
    using FNNLayer<T>::FNNLayer;
};