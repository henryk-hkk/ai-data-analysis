#pragma once

#include "Dataset.h"
#include "ObjectFNN.h"
#include <conio.h>
#include <string>
#include <fstream>
#include <sstream>
#include <direct.h>
#include <ctime>
#include <iomanip>
#include <io.h>

class FileManager { //deprecated - update fnn saving to include layer-dependent activation function saving
public:
    template<arithmetic T>
    static Dataset<T> getDatasetFromCSV(const std::string& filePath) {
        Dataset<T> dataset;
        //std::vector<std::vector<double>> inputVec, outputVec;
        std::ifstream file(filePath);
        if (!file.good()) {
            std::cout << "Provided CSV file is incorrect.\n";
            return Dataset<T>{ {}, {} };
        }

        std::string line;
        std::getline(file, line);//skipping the header

        while (std::getline(file, line)) {
            std::vector<T> row;
            std::stringstream ss(line);
            std::string cell;
            unsigned i{};
            while (std::getline(ss, cell, ',')) {
                if (i == 0) {
                    dataset.desiredOutputVector.push_back(mfuncs::convertDigitToOutputVector<T>(std::stoi(cell)));
                    i++;
                    continue;
                }
                if(std::is_same<T, float>::value) row.push_back(std::stof(cell) / 255.0f);
                else row.push_back(std::stod(cell) / 255.0);
            }
            dataset.inputVector.push_back(row);
        }
        file.close();

        return dataset;
    }

    static std::vector<double> getInputVectorFrom28x28BMP(const std::string& filePath) {
        std::vector<double> inputVec;
        inputVec.reserve(28 * 28);
        std::ifstream file;

        file.open(filePath, std::ios::in | std::ios::binary);

        uint32_t dataOffset{};
        file.seekg(10);
        file.read(reinterpret_cast<char*>(&dataOffset), 4);
        file.seekg(dataOffset);
        //file header + information

        int rowSize = ((24 * 28 + 31) / 32) * 4;
        int padding = rowSize - 28 * 3;

        for (int y{27}; y >= 0; y--) {
            for (int x{}; x < 28; x++) {
                unsigned char r{}, g{}, b{};
                file.read(reinterpret_cast<char*>(&b), 1);
                file.read(reinterpret_cast<char*>(&g), 1);
                file.read(reinterpret_cast<char*>(&r), 1);

                double gray = (r + g + b) / 3.0;

                double mnistPixel = gray;

                inputVec.push_back(mnistPixel);
            }
            file.ignore(padding);
        }
        return inputVec;
    }
    template<arithmetic T>
    static void SaveDatasetToCSV(const Dataset<T> &dataset, const std::string& filePath) {
        std::ofstream file(filePath);
        for (size_t i{}; i < dataset.inputVector.size(); i++) {
            file << mfuncs::convertOutputVectorToDigit(dataset.desiredOutputVector[i]) << ",";
            for (size_t inputPixel{}; inputPixel < dataset.inputVector[i].size(); inputPixel++) {
                file << dataset.inputVector[i][inputPixel] << ",";
            }
            file << "\n";
        }
        file.close();
    }
    static void saveObjectFNN(const ObjectFNN<double>& fnn, std::string savePath) {
        std::tm time = getLocalTime();
        size_t layerCount = fnn.layers.size();
        std::ostringstream oss;
        oss << "/neural_network_" << std::put_time(&time, "%Y-%m-%d_%H-%M-%S");
        savePath += oss.str();

        int status = _mkdir(savePath.c_str());
        if (status != 0) {
            std::cout << "saving failed!";
            return;
        }
        saveFNNInputLayerToCSV(fnn.layers[0], savePath + "/layer0");
        for (size_t i{1}; i < layerCount; i++) {
            saveFNNLayerToCSV(fnn.layers[i], savePath + "/layer" + std::to_string(i));
        }
        saveFNNActivationFunctionsToCSV(fnn, savePath);
    }
    static ObjectFNN<double> loadObjectFNN(std::string loadPath) {
        ObjectFNN<double> fnn;
        //add load path validation
        int fileCount = getFolderFileCount(loadPath);
        
        std::ifstream inputLayer(loadPath + "/layer0_input.csv");
        std::string inputLayerSizeStr;
        std::getline(inputLayer, inputLayerSizeStr);
        size_t inputLayerSize = stoi(inputLayerSizeStr);
        fnn.addInputLayer(inputLayerSize);
        inputLayer.close();

        fileCount -= 2;//ommiting the input layer file and activation functions files
        size_t layerCount = static_cast<size_t>(fileCount / 2) + 1;

        auto funcs = getActivationFunctions(loadPath + "/functions.csv");
        fnn.setLayerActivationFunction(0, funcs[0]);
        for (size_t i{ 1 }; i < layerCount; i++) {
            auto lV = getObjectFNNLayerVectors(loadPath + "/layer" + std::to_string(i));
            size_t neuronCount = lV.biases.size();
            if (i == layerCount - 1) fnn.addOutputLayer(neuronCount);
            else fnn.addLayer(neuronCount);
            fnn.layers[i]->weights = lV.weights;
            for (size_t j{}; j < neuronCount; j++)
                fnn.layers[i]->neurons[j]->bias = lV.biases[j];
            fnn.setLayerActivationFunction(i, funcs[i]);
        }

        return fnn;
    }
private:
    static void saveFNNLayerToCSV(const std::unique_ptr<FNNLayer<double>>& layer, const std::string& filePath) {
        LayerVectors lV = layer->getLayerVectors();
        std::ofstream weights(filePath + "_weights.csv"), biases(filePath + "_biases.csv");
        bool firstLineFlag = true;
        for (size_t neuron = 0; neuron < lV.weights.size(); ++neuron) {
            if (neuron > 0) {
                biases << ",";
                weights << "\n";
            }
            biases << lV.biases[neuron];

            for (size_t prev = 0; prev < lV.weights[neuron].size(); ++prev) {
                if (prev > 0) weights << ",";
                weights << lV.weights[neuron][prev];
            }
        }
        weights.close();
        biases.close();
    }
    static void saveFNNInputLayerToCSV(const std::unique_ptr<FNNLayer<double>>& inputLayer, const std::string& filePath) {
        std::ofstream input(filePath + "_input.csv");
        input << inputLayer->neuronCount;
        input.close();
        return;
    }
    static void saveFNNActivationFunctionsToCSV(const ObjectFNN<double>& fnn, const std::string& savePath) {
        std::ofstream functions(savePath + "/functions.csv");
        for (size_t layer{}; layer < fnn.layers.size(); layer++) {
            functions << static_cast<int>(fnn.layers[layer]->activationFunc) << ",";
        }
        functions.close();
    }
protected:
    static LayerVectors<double> getObjectFNNLayerVectors(const std::string& filePath) {
        std::string line, cell;

        std::vector<double> biases;
        std::vector<std::vector<double>> weights;

        std::ifstream layerBiasFile(filePath + "_biases.csv");
        std::getline(layerBiasFile, line);
        std::stringstream ss(line);

        while (std::getline(ss, cell, ','))
            biases.push_back(std::stod(cell));

        layerBiasFile.close();

        std::vector<double> row;
        std::ifstream layerWeightsFile(filePath + "_weights.csv");
        while (std::getline(layerWeightsFile, line)) {
            std::stringstream ss(line);
            while (std::getline(ss, cell, ','))
                row.push_back(std::stod(cell));
            weights.push_back(row);
            row.clear();
        }
        layerWeightsFile.close();

        LayerVectors lV{ weights, biases };
        return lV;

    }

    static std::vector<AFunc> getActivationFunctions(const std::string& filePath) {
        std::ifstream activationFunctionsFile(filePath);
        std::string line, cell;
        std::vector<AFunc> activationFunctions;
        while (std::getline(activationFunctionsFile, line)) { //file should only contain one line anyway
            std::stringstream ss(line);
            while (std::getline(ss, cell, ','))
                activationFunctions.push_back(static_cast<AFunc>(std::stoi(cell)));
        }
        return activationFunctions;
    }

    static std::tm getLocalTime() {
        std::time_t t = std::time(nullptr);
        std::tm time{};
        localtime_s(&time, &t);
        return time;
    }
    static int getFolderFileCount(const std::string& path) {
        _finddata_t data;
        intptr_t handle = _findfirst((path + "\\*").c_str(), &data);
        int count = (handle != -1);

        while (_findnext(handle, &data) == 0)
            ++count;

        _findclose(handle);
        return count - 2;
    }
};

