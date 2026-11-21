#pragma once
#include <concepts>
template <typename T>
concept arithmetic = std::integral<T> || std::floating_point<T>;