#include "MatrixFNNBackpropA.h"
#include "FileManager.h"
#include "PCAnalyser.h"
#include "Tensor.hpp"

template<typename T>
static void print(const std::vector<T>& vec) {
	for (size_t i{}; i < vec.size(); i++)
		std::cout << vec[i] << "\n";
	return;
}

static void testObjectFnn() {
	//auto dataset = FileManager::getDatasetFromCSV("C:/Users/zimor/Documents/neural networks/mnist_train.csv");
	std::vector<double> elo({ 1,2,3 }), elo2({ 2,3,4 });
	mfuncs::addVectors(elo, elo2);
	ObjectFNN<double> fnn;

	//fnn.addInputLayer(784);
	//fnn.addLayer(16);
	//fnn.addLayer(16);
	//fnn.addOutputLayer(10);
	//fnn.setActivationFunction(ActivationFunction::SIGMOID);
	//fnn.setOutputLayerActivationFunction(ActivationFunction::SOFTMAX);
	// 
	//FileManager::saveLayerToCSV(nn.layers[1], "C:/Users/zimor/Documents/Layer1");
	fnn = FileManager::loadObjectFNN("C:/Users/zimor/Documents/NN_91.5Acc");
	//fnn.setActivationFunction(MathFunctions::sigmoid, MathFunctions::dSigmoid);
	//for (size_t i{}; i < nn.layers.size(); i++) {
	//	std::cout << nn.layers[i]->neuronCount << "\n";
	//}

	//nn.process(dataset.inputVector[0]);


	//FNNTrainingAgent::trainFNN(fnn, dataset, 2, 0.05, true);

	//Dataset dataset = FileManager::getDatasetFromCSV("C:/Users/zimor/Documents/neural networks/mnist_test.csv");
	//std::cout << "\n" << ObjectFNNTrainingAgent::testFNN(fnn, dataset);
	//std::cout << "\n";

	std::string filePath{ "C:/Users/zimor/Pictures/3.bmp" };
	fnn.forward(FileManager::getInputVectorFrom28x28BMP(filePath));
	std::vector<double> outputArr = fnn.getOutput();
	//for (int i = 0; i < 10; i++) {
	//	std::cout << outputArr[i] << "\n";
	//}
	std::cout << "\n" << mfuncs::convertOutputVectorToDigit(outputArr);

	//std::vector<double> desiredOut({0, 0, 0, 1, 0, 0, 0, 0, 0, 0});
	//std::vector<double> input = FileManager::getInputVectorFrom28x28BMP("C:/Users/zimor/Pictures/3.bmp");
	//auto testDataset = Dataset{ {input}, {desiredOut} };
	//FileManager::SaveDatasetToCSV(testDataset, "C:/Users/zimor/Pictures/elo.csv");
	//std::cout << "\n\n" << MathFunctions::convertOutputVectorToDigit(desiredOut);
	//nn.process(inputArr);
	//std::cout << "\n\n";
	//outputArr = nn.getOutput();
	//for (int i = 0; i < 10; i++) {
	//	std::cout << outputArr[i] << "\n";
	//}
	//std::cout << "\n\n" << nn.getCost(desiredOut);

	FileManager::saveObjectFNN(fnn, "C:/Users/zimor/Documents");
}

static void testMatrix() {
	Matrix dataMatrix(3, 3);
	//dataMatrix.setContent({ {1,1,1}, {1, 2,1}, {1,1,2} });
	dataMatrix.fillWithRandomIntegers(1, 10);
	std::cout << dataMatrix;
	dataMatrix = mfuncs::getMeanCenteredMatrix(dataMatrix);
	std::cout << dataMatrix;
	auto bidiagonal = SVDAgent<double>::bidiagonalize(dataMatrix);
	std::cout << "\n";
	std::cout << bidiagonal.B;
	std::cout << "\n";
	std::cout << bidiagonal.U1;
	std::cout << "\n";
	std::cout << bidiagonal.V1_T;
	std::cout << "\n";
	//SVDAgent::getSVD(dataMatrix).print();
	std::cout << ((bidiagonal.U1 * bidiagonal.B) * bidiagonal.V1_T);
	std::cout << "\n";
	std::cout << dataMatrix;
	dataMatrix.fill();
	std::cout << "\n";
	std::cout << dataMatrix;
	//auto dV = SVDAgent::getBidiagonalVectors(bidiagonal);
	//print(dV.diagonal);
	//printVector(dV.superDiagonal);
}

static void testMatrixFNN() {
	auto dataset = FileManager::getDatasetFromCSV<float>("Datasets/mnist_train.csv");

	MatrixFNN<float> fnn;

	fnn.addInputLayer(784);
	fnn.addLayer(16);
	fnn.addLayer(16);
	fnn.addOutputLayer(10);
	fnn.setActivationFunction(AFunc::RELU);
	fnn.setOutputLayerActivationFunction(AFunc::SOFTMAX);

	MatrixFNNBackpropA::trainFNN(fnn, dataset, 3, 0.05, 128);

	dataset = FileManager::getDatasetFromCSV<float>("Datasets/mnist_test.csv");
	std::cout << "\n" << MatrixFNNBackpropA::testFNN(fnn, dataset);
}
static void testTensor() {
	Tensor<float, 3> t1({ 2,2,2 });
	//for (size_t i{}; i < t1.dimension(); i++) {
	//	for (size_t j{}; j < t1.dimensionShifts[i].size(); j++) {
	//		std::cout << t1.dimensionShifts[i][j] << "\t";
	//	}
	//	std::cout << "\n";
	//}
}

int main() {
	testTensor();
}