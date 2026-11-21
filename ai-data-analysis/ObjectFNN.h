#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include "Concepts.h"
#include "FNNLayers.h"
#include "MatrixFNN.h"

template<typename T = double>
    requires arithmetic<T>
class ObjectFNN { // Object-based Feedforward Neural Network
public:
    std::vector<std::unique_ptr<FNNLayer<T>>> layers;

    void setActivationFunction(ActivationFunction func) {
        for (size_t i = 0; i + 1 < layers.size(); ++i) { // bez warstwy wyjściowej
            layers[i]->activationFunc = func;
        }
    }

    void setLayerActivationFunction(size_t layer, ActivationFunction func) {
        layers.at(layer)->activationFunc = func;
    }

    void setOutputLayerActivationFunction(ActivationFunction func) {
        if (layers.empty()) {
            throw std::logic_error("Cannot set output activation: network has no layers.");
        }
        layers.back()->activationFunc = func;
    }

    void addInputLayer(size_t layerSize) {
        layers.clear();
        layers.push_back(std::make_unique<FNNInputLayer<T>>(layerSize));
    }

    void addOutputLayer(size_t layerSize) {
        if (layers.empty()) {
            throw std::logic_error("Add an input layer before adding an output layer.");
        }
        FNNLayer<T>* lastLayerPtr = layers.back().get();
        layers.push_back(std::make_unique<FNNOutputLayer<T>>(layerSize, lastLayerPtr));
        lastLayerPtr->next = layers.back().get();
    }

    void addLayer(size_t layerSize) {
        if (layers.empty()) {
            throw std::logic_error("Add an input layer before adding a hidden layer.");
        }
        FNNLayer<T>* lastLayerPtr = layers.back().get();
        layers.push_back(std::make_unique<FNNHiddenLayer<T>>(layerSize, lastLayerPtr));
        lastLayerPtr->next = layers.back().get();
    }

    void process(const std::vector<T>& inputArr) {
        if (layers.empty()) {
            std::cerr << "NETWORK HAS NO LAYERS\n";
            return;
        }

        auto* inputLayerPtr = dynamic_cast<FNNInputLayer<T>*>(layers[0].get());
        if (!inputLayerPtr) {
            std::cerr << "NOT AN INPUT LAYER\n";
            return;
        }

        inputLayerPtr->processInput(inputArr);
        for (size_t i = 1; i < layers.size(); ++i) {
            layers[i]->processInput();
        }

        if (layers.back()->activationFunc == ActivationFunction::SOFTMAX) {
            auto output = mfuncs::softmax(getOutput());
            for (size_t i = 0; i < layers.back()->neuronCount; ++i) {
                layers.back()->neurons[i]->value = output[i];
            }
        }
    }

    [[nodiscard]] std::vector<T> getOutput() const {
        if (layers.empty()) {
            return {};
        }

        const size_t outputLayerSize = layers.back()->neuronCount;
        std::vector<T> output(outputLayerSize);
        for (size_t i = 0; i < outputLayerSize; ++i) {
            output[i] = layers.back()->neurons[i]->value;
        }
        return output;
    }

    [[nodiscard]] T getCost(const std::vector<T>& desiredOutput) const {
        const std::vector<T> output = getOutput();
        if (desiredOutput.size() != output.size()) {
            throw std::invalid_argument("Desired output size does not match network output size.");
        }

        T cost{};
        for (size_t i = 0; i < output.size(); ++i) {
            const T diff = desiredOutput[i] - output[i];
            cost += diff * diff;
        }
        return cost;
    }

    template<MatrixLayout L = MatrixLayout::RowMajor>
    [[nodiscard]] MatrixFNN<T, L> toMatrixForm() const {
        if (layers.empty()) {
            throw std::logic_error("Cannot convert an empty network to matrix form.");
        }

        const auto* inputPtr = dynamic_cast<const FNNInputLayer<T>*>(layers[0].get());
        if (!inputPtr) {
            throw std::logic_error("First layer is not an FNNInputLayer.");
        }

        MatrixFNN<T, L> form;
        form.inputLayer = inputPtr->toMatrixForm();
        form.layers.reserve(layers.size() - 1);

        for (size_t i = 1; i < layers.size(); ++i) {
            form.layers.push_back(layers[i]->template toMatrixForm<L>());
        }

        return form;
    }
};