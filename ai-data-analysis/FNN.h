#pragma once
#include "FNNLayers.h"
#include <iostream>
class FNN { //Feedforward Neural Network
public:
	std::vector<std::unique_ptr<FNNLayer>> layers;
	void setActivationFunction(ActivationFunction func);
	void setLayerActivationFunction(size_t layer, ActivationFunction func);
	void setOutputLayerActivationFunction(ActivationFunction func);
	void addInputLayer(size_t layerSize);
	void addOutputLayer(size_t layerSize);
	void addLayer(size_t layerSize);
	void process(const std::vector<double>& inputArr);
	
	std::vector<double> getOutput() const;
	double getCost(const std::vector<double>& desiredOutput) const;
};