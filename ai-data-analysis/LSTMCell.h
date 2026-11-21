#pragma once
#include <vector>
#include "FNN.h"

struct LSTMCache {
	std::vector<std::vector<double>> combined_history;
	std::vector<std::vector<double>> fg_history;
	std::vector<std::vector<double>> ig_history;
	std::vector<std::vector<double>> og_history;
	std::vector<std::vector<double>> candidate_history;
	std::vector<std::vector<double>> prevCellState_history;
	std::vector<std::vector<double>> cellState_history;
};

struct LSTMGate {
	FNN fnn;
	LSTMGate(size_t inputSize, size_t outputSize, ActivationFunction func) {
		fnn.addInputLayer(inputSize);
		fnn.addOutputLayer(outputSize);
		fnn.setActivationFunction(func);
	};
	
	std::vector<double> process(const std::vector<double>& input) {
		fnn.process(input);
		return fnn.getOutput();
	}
	virtual ~LSTMGate() = default;
};

struct LSTMCell {
	LSTMCell(size_t inputDim, size_t hiddenDim) :
		fg(hiddenDim + inputDim, hiddenDim, ActivationFunction::SIGMOID),
		ig(hiddenDim + inputDim, hiddenDim, ActivationFunction::SIGMOID),
		og(hiddenDim + inputDim, hiddenDim, ActivationFunction::SIGMOID),
		c(hiddenDim + inputDim, hiddenDim, ActivationFunction::TANH)
	{ 
		cellState = std::vector<double>(hiddenDim, 0.0);
		hiddenState = std::vector<double>(hiddenDim, 0.0);
	}

	LSTMGate fg, ig, og, c; //forget, input, output gates and candidate gate.
	LSTMCache cache{};
	std::vector<double> cellState, hiddenState;

	void activate(const std::vector<double>& input, bool training = false) {
		if (training) cache.prevCellState_history.push_back(cellState); //saving the initial cell state

		auto combined = mfuncs::combineVectors(hiddenState, input);
		if (training) cache.combined_history.push_back(combined);

		auto fgOutput = fg.process(combined);
		if (training) cache.fg_history.push_back(fgOutput);

		cellState = mfuncs::hadamard(fgOutput, cellState);

		auto igOutput = ig.process(combined);
		if (training) cache.ig_history.push_back(igOutput);

		auto candidate = c.process(combined);
		if (training) cache.candidate_history.push_back(candidate);

		cellState = mfuncs::addVectors(cellState, mfuncs::hadamard(igOutput, candidate));
		if (training) cache.cellState_history.push_back(cellState);

		auto ogOutput = og.process(combined);
		if (training) cache.og_history.push_back(ogOutput);

		hiddenState = mfuncs::hadamard(ogOutput, mfuncs::processVector(cellState, mfuncs::tanh));
	}
};