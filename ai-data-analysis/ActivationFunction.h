#pragma once
#include "mfuncs.h"
enum class ActivationFunction {
	SIGMOID,
	TANH,
	RELU,
	SOFTMAX,
	LINEAR,
};

static auto getActivationFunction(ActivationFunction func) -> double(*)(double) {
	switch (func) {
	case ActivationFunction::SIGMOID:	return mfuncs::sigmoid;
	case ActivationFunction::TANH:		return mfuncs::tanh;
	case ActivationFunction::RELU:		return mfuncs::ReLU;
	default:      return mfuncs::linear;
	}
}

static auto getDerivative(ActivationFunction func) -> double(*)(double) {
	switch (func) {
	case ActivationFunction::SIGMOID:	return mfuncs::dSigmoid;
	case ActivationFunction::TANH:		return mfuncs::dTanh;
	case ActivationFunction::RELU:		return mfuncs::dReLU;
	default:      return mfuncs::linear;
	}
}