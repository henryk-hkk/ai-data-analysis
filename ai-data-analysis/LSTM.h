#pragma once
#include "LSTMCell.h"
class LSTM { //Long short-term memory Neural Network
	std::vector<std::unique_ptr<LSTMCell>> cells;
	std::unique_ptr<FNN> classificationLayer;

	void addCell(size_t hiddenDim, size_t inputDim = 0) {
		if (cells.empty()) {
			cells.push_back(std::make_unique<LSTMCell>(inputDim, hiddenDim));
		}
		else {
			size_t previousHiddenDim = cells.back()->hiddenState.size();
			cells.push_back(std::make_unique<LSTMCell>(previousHiddenDim, hiddenDim));
		}
	}

	void addClassificationLayer(size_t inputDim, size_t outputDim) {
		classificationLayer = std::make_unique<FNN>();
		classificationLayer->addInputLayer(inputDim);
		classificationLayer->addOutputLayer(outputDim);
	}

	std::vector<double> process(const std::vector<std::vector<double>>& inputVec) {
		if (cells.empty() || inputVec.empty()) return {};

		for (size_t t{}; t < inputVec.size(); t++) {
			cells[0]->activate(inputVec[t]);
			for (size_t i{1}; i < cells.size(); i++)
				cells[i]->activate(cells[i - 1]->hiddenState);
		}
		return cells.back()->hiddenState;
	}

	void train(const std::vector<std::vector<double>>& inputVec, const std::vector<std::vector<double>>& outputVec) {
		if (cells.empty() || inputVec.empty() || outputVec.empty()) return;
		for (size_t t{}; t < inputVec.size(); t++) {
			cells[0]->activate(inputVec[t], true);
			for (size_t i{ 1 }; i < cells.size(); i++)
				cells[i]->activate(cells[i - 1]->hiddenState, true);
		}
	}
};