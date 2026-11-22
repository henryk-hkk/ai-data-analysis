#include "mfuncs.h"
std::vector<double> mfuncs::softmax(const std::vector<double>& vec) {
	double sum{};
	std::vector<double> output(vec.size());
	auto max = mfuncs::max(vec);

	for (size_t i{}; i < vec.size(); i++) {
		output[i] = std::exp(vec[i] - max); // to avoid overflow
		sum += output[i];
	}
	for (size_t i{}; i < vec.size(); i++)
		output[i] /= sum;
	return output;
}

double mfuncs::getRandomDouble(double a, double b) {
	return a + (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * (b - a);
}

int mfuncs::getRandomInteger(int a, int b) {
	return a + (rand() % (b - a + 1));
}
std::vector<double> mfuncs::combineVectors(const std::vector<double>& vec1, const std::vector<double>& vec2) {
	std::vector<double> result(vec1.size() + vec2.size());
	for (size_t i{}; i < vec1.size(); i++)
		result[i] = vec1[i];
	for (size_t i{}; i < vec2.size(); i++)
		result[i + vec1.size()] = vec2[i];
	return result;
}

std::vector<double> mfuncs::addVectors(const std::vector<double>& vec1, const std::vector<double>& vec2) {
	if (vec1.size() != vec2.size()) return std::vector<double>();
	std::vector<double> result(vec1.size());
	for (size_t i{}; i < vec1.size(); i++)
		result[i] = vec1[i] + vec2[i];
	return result;
}

std::vector<double> mfuncs::processVector(const std::vector<double>& input, double (*function)(double x)) {
	std::vector<double> output(input.size());
	for (size_t i{}; i < input.size(); i++)
		output[i] = function(input[i]);
	return output;
}

std::vector<double> mfuncs::hadamard(const std::vector<double>& vec1, const std::vector<double>& vec2) {
	if (vec1.size() != vec2.size()) return std::vector<double>();
	std::vector<double> hadamardProduct(vec1.size());
	for (size_t i{}; i < vec1.size(); i++)
		hadamardProduct[i] = vec1[i] * vec2[i];
	return hadamardProduct;
}