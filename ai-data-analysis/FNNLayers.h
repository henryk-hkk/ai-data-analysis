#pragma once
#include "Neuron.h"
#include "mfuncs.h"
#include "ActivationFunction.h"
#include <memory>
#include <ctime>

class FNNLayer {
public:
	FNNLayer() = default;
	FNNLayer(size_t neuronCount, ActivationFunction func = ActivationFunction::LINEAR) : neuronCount(neuronCount), activationFunc(func), activations(neuronCount) {}
	FNNLayer(size_t neuronCount, FNNLayer* prev, ActivationFunction func = ActivationFunction::LINEAR) : neuronCount(neuronCount), previous(prev), activationFunc(func), activations(neuronCount) { init(); }

	void init();

	void processInput();
	FNNLayer* previous = nullptr, * next = nullptr;
	size_t neuronCount = 0;
	std::vector<std::vector<double>> weights;//weights[i][j] -> i-index of current neuron, j-indexes of previous layer's neurons
	std::vector<std::unique_ptr<Neuron>> neurons;
	std::vector<double> activations;
	ActivationFunction activationFunc;
	virtual ~FNNLayer() = default;
};

struct FNNInputLayer : FNNLayer {
	FNNInputLayer(size_t neuronCount, ActivationFunction func = ActivationFunction::LINEAR);
	void processInput(std::vector<double> inputArr);
};

struct FNNHiddenLayer : FNNLayer {
	using FNNLayer::FNNLayer;
};

struct FNNOutputLayer : FNNLayer {
	using FNNLayer::FNNLayer;
};