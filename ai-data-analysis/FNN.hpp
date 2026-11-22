#pragma once
#include "AFunc.h"
enum class WeightInit {
	Xavier,  // Sigmoid, Tanh, Linear, Softmax
	He,      // ReLU, LeakyReLU
	Uniform
};

template<typename T>
    requires arithmetic<T>
class FNN { // Feedforward Neural Network
protected:
    FNN() = default;
    FNN(const FNN&) = default;
    FNN& operator=(const FNN&) = default;
    FNN(FNN&&) noexcept = default;
    FNN& operator=(FNN&&) noexcept = default;

public:
    virtual ~FNN() = default;

    virtual void setActivationFunction(AFunc func) = 0;
    virtual void setLayerActivationFunction(size_t layerIndex, AFunc func) = 0;
    virtual void setOutputLayerActivationFunction(AFunc func) = 0;

    virtual void addInputLayer(size_t layerSize, AFunc func = AFunc::LINEAR) = 0;
    // virtual void addLayer(size_t layerSize, AFunc func = AFunc::LINEAR, WeightInit init = WeightInit::Xavier) = 0;
    // virtual void addOutputLayer(size_t layerSize, AFunc func = AFunc::SOFTMAX, WeightInit init = WeightInit::Xavier) = 0;
    virtual std::vector<T> forward(const std::vector<T>& input) = 0;
};