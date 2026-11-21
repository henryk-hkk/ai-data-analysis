#pragma once
#include "Concepts.h"

template<typename T = double>
	requires arithmetic<T>
struct Neuron {
	T value{1}, bias{};
 
	void activate(T input) { value = input; }

	Neuron(T bias = 0) : bias(bias) {};
};