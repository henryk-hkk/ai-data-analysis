#pragma once

#include <iostream>
#include <type_traits>
#include <vector>
#include "Concepts.h"
#include "ObjectFNN.h"
#include "Dataset.h"

class ObjectFNNTrainingAgent {
public:
    template<typename T, typename DatasetT>
        requires arithmetic<T>
    static void trainFNN(
        ObjectFNN<T>& fnn,
        const DatasetT& trainingDataset,
        unsigned epochs,
        std::type_identity_t<T> learningFactor,
        bool getConsoleOutput = false
    ) {
        std::vector<std::vector<std::vector<T>>> errorsArr;
        std::vector<std::vector<T>> errorsMean;
        const size_t inputsPerIteration = 1;
        const size_t iterations = trainingDataset.inputVector.size() / inputsPerIteration;

        for (unsigned epoch = 0; epoch < epochs; ++epoch) {
            for (size_t iteration = 0; iteration < iterations; ++iteration) {
                errorsArr.clear();
                errorsArr.resize(inputsPerIteration);

                for (size_t i = 0; i < inputsPerIteration; ++i) {
                    const size_t sampleIdx = iteration * inputsPerIteration + i;
                    fnn.process(trainingDataset.inputVector[sampleIdx]);
                    errorsArr[i] = getNeuronErrors(fnn, trainingDataset.desiredOutputVector[sampleIdx]);
                }

                errorsMean = getErrorsMean(fnn, errorsArr, inputsPerIteration);
                adjustFNNValues(fnn, errorsMean, learningFactor);

                if (getConsoleOutput) {
                    const size_t lastIdx = (iteration + 1) * inputsPerIteration - 1;
                    std::cout << "\nTraining iteration: " << (iteration + 1)
                        << ". Iteration cost: " << fnn.getCost(trainingDataset.desiredOutputVector[lastIdx]);
                }
            }
        }
    }

    template<typename T, typename DatasetT>
        requires arithmetic<T>
    [[nodiscard]] static double testFNN(ObjectFNN<T>& fnn, const DatasetT& testDataset) {
        if (testDataset.inputVector.empty()) {
            return 0.0;
        }

        size_t correct = 0;
        for (size_t i = 0; i < testDataset.inputVector.size(); ++i) {
            fnn.process(testDataset.inputVector[i]);
            if (mfuncs::convertOutputVectorToDigit(fnn.getOutput()) ==
                mfuncs::convertOutputVectorToDigit(testDataset.desiredOutputVector[i])) {
                ++correct;
            }
        }

        return static_cast<double>(correct) / static_cast<double>(testDataset.inputVector.size());
    }

protected:
    template<typename T>
        requires arithmetic<T>
    static std::vector<std::vector<T>> getErrorsMean(
        const ObjectFNN<T>& fnn,
        const std::vector<std::vector<std::vector<T>>>& errorsArr,
        size_t inputsPerIteration
    ) {
        std::vector<std::vector<T>> errorsMean(fnn.layers.size());
        for (size_t layer = 1; layer < fnn.layers.size(); ++layer) {
            errorsMean[layer].resize(fnn.layers[layer]->neuronCount, T{ 0 });
            for (size_t neuron = 0; neuron < fnn.layers[layer]->neuronCount; ++neuron) {
                for (size_t i = 0; i < inputsPerIteration; ++i) {
                    errorsMean[layer][neuron] += errorsArr[i][layer][neuron];
                }
                errorsMean[layer][neuron] /= static_cast<T>(inputsPerIteration);
            }
        }
        return errorsMean;
    }

    template<typename T>
        requires arithmetic<T>
    static std::vector<std::vector<T>> getNeuronErrors(
        const ObjectFNN<T>& fnn,
        const std::vector<T>& desiredOutput
    ) {
        if (fnn.layers.size() < 2) {
            return {};
        }

        std::vector<std::vector<T>> errors(fnn.layers.size());

        for (size_t layer = fnn.layers.size() - 1; layer > 0; --layer) {
            errors[layer].resize(fnn.layers[layer]->neuronCount);
            const auto derivFunc = getDerivative(fnn.layers[layer]->activationFunc);

            for (size_t neuron = 0; neuron < fnn.layers[layer]->neuronCount; ++neuron) {
                const T currentNeuronValue = fnn.layers[layer]->neurons[neuron]->value;

                if (layer == fnn.layers.size() - 1) {
                    // dla softmax + cross-entropy
                    errors[layer][neuron] = currentNeuronValue - desiredOutput[neuron];
                }
                else {
                    T nextLayerErrorSum{};
                    for (size_t nextLayerNeuron = 0; nextLayerNeuron < fnn.layers[layer + 1]->neuronCount; ++nextLayerNeuron) {
                        nextLayerErrorSum += fnn.layers[layer + 1]->weights[nextLayerNeuron][neuron] * errors[layer + 1][nextLayerNeuron];
                    }
                    errors[layer][neuron] = nextLayerErrorSum * static_cast<T>(derivFunc(static_cast<double>(currentNeuronValue)));
                }
            }
        }

        return errors;
    }

    template<typename T>
        requires arithmetic<T>
    static void adjustFNNValues(
        ObjectFNN<T>& fnn,
        const std::vector<std::vector<T>>& errors,
        std::type_identity_t<T> learningFactor
    ) {
        for (size_t layer = 1; layer < fnn.layers.size(); ++layer) {
            for (size_t neuron = 0; neuron < fnn.layers[layer]->neuronCount; ++neuron) {
                fnn.layers[layer]->neurons[neuron]->bias -= learningFactor * errors[layer][neuron];
                for (size_t previousLayerNeuron = 0; previousLayerNeuron < fnn.layers[layer - 1]->neuronCount; ++previousLayerNeuron) {
                    fnn.layers[layer]->weights[neuron][previousLayerNeuron] -=
                        learningFactor * errors[layer][neuron] * fnn.layers[layer - 1]->neurons[previousLayerNeuron]->value;
                }
            }
        }
    }
};