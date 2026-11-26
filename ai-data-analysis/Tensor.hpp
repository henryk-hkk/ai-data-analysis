#pragma once
#include <vector>
#include <stdexcept>
#include "Concepts.h"
#include "Matrix.h"
template<typename T, size_t Dimension>
	requires arithmetic<T>
class Tensor {
private:
	std::vector<T> data;
	std::vector<size_t> dimensionSizes;
	// [0][4] - shift to dimesion 0's index 4 start from data vector beginning, corresponding to T[4]. Sum of shifts [0][4] and [1][2] would correspond to T[4][2], etc.
	// meant to speed up the indexing process instead of multiplying e.g. index = x4 + x3 * dimensionSize[2] + x2 * dimensionSize[1] + x1 * dimensionSize[0]
	// will test if it really speeds it up, probably not because of cache misses
	std::vector<std::vector<size_t>> dimensionShifts;
	static std::vector<std::vector<size_t>> getDimensionShifts(std::vector<size_t> dimensionSizes) {
		size_t dim = dimensionSizes.size();
		std::vector<std::vector<size_t>> shifts(dim);
		for (size_t i{}; i < dim; i++) {
			shifts[i].resize(dimensionSizes[i]);

			size_t dimensionBlockSize{ 1 };
			for (size_t j{ i }; j < dim; j++)
				dimensionBlockSize *= dimensionSizes[j];

			for (size_t j{}; j < dimensionSizes[i]; j++)
				shifts[i][j] = j * dimensionBlockSize;
		}
		return shifts;
	}
	constexpr size_t index(std::initializer_list<size_t> indices) const {
		if (indices.size() > Dimension) throw new std::invalid_argument("");
		size_t index{};
		for (size_t i{}; i < indices.size(); i++)
			index += dimensionShifts[i][indices.data()[i]];
		return index;
	}
public:
	Tensor() {}
	Tensor(std::initializer_list<size_t> dimensionSizes) : dimensionSizes(dimensionSizes) {
		if (dimensionSizes.size() != Dimension) throw new std::invalid_argument("Dimension sizes array length must match provided tensor dimension");
		size_t size{ 1 };
		for (size_t i{}; i < Dimension; i++)
			size *= dimensionSizes.data()[i];
		data.resize(size);
		dimensionShifts = getDimensionShifts(dimensionSizes);
	}
	Tensor(std::initializer_list<size_t> dimensionSizes, std::vector<T> values) : dimensionSizes(dimensionSizes), data(values) {
		if (dimensionSizes.size() != Dimension) throw new std::invalid_argument("");
		if (size() != values.size()) throw new std::invalid_argument("");

		dimensionShifts = getDimensionShifts(dimensionSizes);
	}

	template<arithmetic U, MatrixLayout L>
	Tensor<U, 2> fromMatrix(Matrix <U, L> m) {
		if (L != MatrixLayout::RowMajor) throw new std::invalid_argument("");
		return Tensor({ m.getWidth(), m.getHeight() }, m.getContent());
	}

	constexpr size_t elemIndex(std::initializer_list<size_t> indices) const { // for individual elements
		if (indices.size() != Dimension) throw new std::invalid_argument("");
		return index(indices);
	}
	constexpr size_t blockIndex(std::initializer_list<size_t> indices) const { // for memory blocks
		if (indices.size() >= Dimension) throw new std::invalid_argument("");
		return index(indices);
	}
	size_t size() const {
		size_t size{ 1 };
		for (size_t i{}; i < Dimension; i++)
			size *= dimensionSizes.data()[i];
		return size;
	}
	size_t dimension() const {
		return Dimension;
	}
	T at(size_t index) const {
		if (index > data.size()) throw new std::invalid_argument("");
		return data[index];
	}
	T at(std::initializer_list<size_t> indices) const {
		return data[elemIndex(indices)];
	}
	template<size_t Dimension2>
	Tensor<T, Dimension2> blockAt(std::initializer_list<size_t> indices) {
		if constexpr (indices.size() > Dimension) throw new std::invalid_argument("");
		size_t newDim = Dimension - indices.size();
		std::vector<size_t> newDimSizes(newDim);
		size_t newDataLen = 1;
		for (size_t i{ newDim }; i < Dimension; i++) {
			size_t dimSize = dimensionSizes[newDim + i];
			newDimSizes[i] = dimSize;
			newDataLen *= dimSize;
		}
		std::vector<T> newData(newDataLen);
		const size_t copyBlockIndex = blockIndex(indices);
		for (size_t i{}; i < newDataLen; i++)
			newData[i] = data[i + copyBlockIndex];
		
		return new Tensor<T, newDim>(newDimSizes, newData);
	}
};