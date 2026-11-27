#pragma once
#include <concepts>
template <typename T>
concept arithmetic = std::integral<T> || std::floating_point<T>;
template <typename T>
concept gpu_compatible = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, int>;