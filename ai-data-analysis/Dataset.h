#pragma once
#include <vector>
template<arithmetic T>
struct Dataset {
	std::vector<std::vector<T>> inputVector{}, desiredOutputVector{};
};