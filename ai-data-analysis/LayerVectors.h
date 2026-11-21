#pragma once
#include <vector>
#include "Concepts.h"
template<typename T>
    requires arithmetic<T>
struct LayerVectors {
    std::vector<std::vector<T>> weights;
    std::vector<T> biases;
};