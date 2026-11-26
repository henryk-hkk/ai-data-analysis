#pragma once
#include <vector>
#include <stdexcept>
#include "Concepts.h"
#include "Matrix.hpp"

template<typename T>
	requires arithmetic<T>
class Tensor { // Row-Major ML Tensor
private:
	std::vector<T> data_;
	std::vector<size_t> shape_{}; // zamiast dimensionSizes. To samo
	std::vector<size_t> strides_{}; // zamiast shifts. Określa długość danego bloku

	static std::vector<size_t> computeStrides(const std::vector<size_t>& shape) {
		if (shape.empty()) return {};
		std::vector<size_t> strides(shape.size(), 1);
		for (size_t i = shape.size() - 1; i > 0; --i) {
			strides[i - 1] = strides[i] * shape[i];
		}
		return strides;
	}
	static size_t computeSize(const std::vector<size_t>& shape) noexcept {
		if (shape.empty()) return 0;
		size_t total = 1;
		for (size_t s : shape) total *= s;
		return total;
	}
	size_t offset(std::span<const size_t> indices) const {
		if (indices.size() > shape_.size()) {
			throw std::invalid_argument("");
		}
		size_t offset = 0;
		for (size_t i = 0; i < indices.size(); ++i) {
			if (indices[i] >= shape_[i]) {
				throw std::out_of_range("");
			}
			offset += indices[i] * strides_[i];
		}
		return offset;
	}
public:
	Tensor() = default;

	Tensor(std::vector<size_t> shape)
		: shape_(std::move(shape)),
		strides_(computeStrides(shape_)),
		data_(computeSize(shape_)) {}

	Tensor(std::vector<size_t> shape, std::vector<T> values)
		: shape_(std::move(shape)),
		strides_(computeStrides(shape_)),
		data_(std::move(values))
	{
		if (computeSize(shape_) != data_.size())
			throw std::invalid_argument("");
	}
	template<MatrixLayout L>
	explicit Tensor(const Matrix<T, L>& m) {
		if constexpr (L == MatrixLayout::ColumnMajor) {
			auto t = m.transposed();
			shape_ = { t.getHeight(), t.getWidth() };
			data_ = t.getContent();
		}
		else {
			shape_ = { m.getHeight(), m.getWidth() };
			data_ = m.getContent();
		}
		strides_ = computeStrides(shape_);
	}
	size_t elemIndex(std::initializer_list<size_t> indices) const { // for individual elements
		if (indices.size() != shape_.size()) throw new std::invalid_argument("");
		return offset({ indices.begin(), indices.size() });
	}
	size_t blockIndex(std::initializer_list<size_t> indices) const { // for memory blocks
		if (indices.empty() || indices.size() >= shape_.size()) throw new std::invalid_argument("");
		return offset({ indices.begin(), indices.size() });
	}

	T& at(size_t index) {
		if (index > data_.size()) throw new std::out_of_range("");
		return data_[index];
	}
	const T& at(size_t index) const {
		if (index > data_.size()) throw new std::out_of_range("");
		return data_[index];
	}

	T& at(std::initializer_list<size_t> indices) {
		return data_[elemIndex(indices)];
	}
	const T& at(std::initializer_list<size_t> indices) const {
		return data_[elemIndex(indices)];
	}
	
	template<std::convertible_to<size_t>... Indices>
	T& operator()(Indices... indices) {
		std::array<size_t, sizeof...(Indices)> idx{ static_cast<size_t>(indices)... };
		if (idx.size() != shape_.size()) {
			throw std::invalid_argument("Index count must match tensor dimension");
		}
		return data_[computeOffset(idx)];
	}

	template<std::convertible_to<size_t>... Indices>
	const T& operator()(Indices... indices) const {
		std::array<size_t, sizeof...(Indices)> idx{ static_cast<size_t>(indices)... };
		if (idx.size() != shape_.size()) {
			throw std::invalid_argument("Index count must match tensor dimension");
		}
		return data_[computeOffset(idx)];
	}

	Tensor<T> blockAt(std::initializer_list<size_t> indices) const {
		const size_t offset = blockIndex(indices);
		const size_t k = indices.size();
		std::vector<size_t> newShape(shape_.begin() + k, shape_.end());
		const size_t blockLen = strides_[k - 1];

		std::vector<T> newData(data_.begin() + offset, data_.begin() + offset + blockLen);
		return Tensor<T>(std::move(newShape), std::move(newData));
	}

	std::span<T> vectorAt(std::initializer_list<size_t> indices) {
		if (shape_.empty() || indices.size() != shape_.size() - 1) {
			throw std::invalid_argument("Vector index count must equal dimension - 1");
		}
		const size_t offset = offset({ indices.begin(), indices.size() });
		return std::span<T>(data_.data() + offset, shape_.back());
	}

	void reshape(std::vector<size_t> newShape) {
		if (computeSize(newShape) != data_.size()) {
			throw std::invalid_argument("New shape total size must match current element count");
		}
		shape_ = std::move(newShape);
		strides_ = computeStrides(shape_);
	}

	const std::vector<T>& getContent() const noexcept { return data_; }
	const std::vector<size_t>& shape() const noexcept { return shape_; }
	const std::vector<size_t>& strides() const noexcept { return strides_; }
	size_t size() const noexcept { return data_.size(); }
	size_t dimension() const noexcept { return shape_.size(); }
};