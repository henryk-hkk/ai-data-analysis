#pragma once
#include "mfuncs.h"
enum class AFunc {
	SIGMOID,
	TANH,
	RELU,
	LEAKYRELU,
	SOFTMAX,
	LINEAR,
};

static auto getActivationFunction(AFunc func) -> double(*)(double) {
	switch (func) {
	case AFunc::SIGMOID:	return mfuncs::sigmoid;
	case AFunc::TANH:		return mfuncs::tanh;
	case AFunc::RELU:		return mfuncs::ReLU;
	case AFunc::LEAKYRELU:	return mfuncs::lReLU;
	default:      return mfuncs::linear;
	}
}

static auto getDerivative(AFunc func) -> double(*)(double) {
	switch (func) {
	case AFunc::SIGMOID:	return mfuncs::dSigmoid;
	case AFunc::TANH:		return mfuncs::dTanh;
	case AFunc::RELU:		return mfuncs::dReLU;
	case AFunc::LEAKYRELU:	return mfuncs::dLReLU;
	default:      return mfuncs::linear;
	}
}