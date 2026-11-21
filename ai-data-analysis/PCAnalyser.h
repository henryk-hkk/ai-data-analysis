#pragma once
#include "SVDAgent.h"
template <typename T>
class PCAnalyser {
public:
	static Matrix<T> getCovarianceMatrix(const Matrix<T>& m) {
		Matrix<T> B = mfuncs::getMeanCenteredMatrix(m);
		return B;
		Matrix<T> C = B.transposed() * B;
		//C = MathFunctions::multiply(C, 1.0 / (B.getHeight() - 1));
		//return C;
	}
//protected:
};