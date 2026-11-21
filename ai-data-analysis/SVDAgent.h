#pragma once

#include "mfuncs.h"
#include <vector>
#include <algorithm>

template <typename T = double>
struct SVD {
    Matrix<T> U, Sigma, V_T;
};

template <typename T = double>
struct BidiagonalForm {
    Matrix<T> U1, B, V1_T;
};

template <typename T = double>
struct BidiagonalVectors {
    std::vector<T> diagonal, superDiagonal;
};

template <typename T = double>
struct BidiagonalEcon {
    Matrix<T> B;
    std::vector<std::vector<T>> U_reflectors;
    std::vector<std::vector<T>> V1_reflectors;
};

template <typename T = double>
struct GivensRotation {
    T cos, sin, norm;
};

template <typename T = double>
class SVDAgent {
public:
    [[nodiscard]] static Matrix<T> getSVD(const Matrix<T>& m) {
        Matrix<T> mCentered = mfuncs::getMeanCenteredMatrix(m);
        auto bidiagonal = bidiagonalize(mCentered);

        Matrix<T> Sigma = bidiagonal.B * bidiagonal.B.transposed();
        Sigma = mfuncs::getSqrtValues(Sigma);

        return Sigma;
    }

    [[nodiscard]] static BidiagonalVectors<T> getBidiagonalVectors(const BidiagonalForm<T>& bidiagonal) {
        return getBidiagonalVectors(bidiagonal.B);
    }

    [[nodiscard]] static BidiagonalVectors<T> getBidiagonalVectors(const Matrix<T>& B) {
        size_t n = std::min(B.getWidth(), B.getHeight());

        BidiagonalVectors<T> bV;
        if (n == 0) return bV;

        bV.diagonal.resize(n);
        bV.superDiagonal.resize(n - 1);

        for (size_t i = 0; i < n; i++) {
            bV.diagonal[i] = B.getValue(i, i);
            if (i < n - 1) {
                bV.superDiagonal[i] = B.getValue(i + 1, i);
            }
        }
        return bV;
    }

    [[nodiscard]] static BidiagonalEcon<T> bidiagonalizeEcon(const Matrix<T>& m) {
        Matrix<T> B = m;
        size_t width = m.getWidth(), height = m.getHeight();

        std::vector<std::vector<T>> U_reflectors;
        std::vector<std::vector<T>> V1_reflectors;

        size_t limit = std::min(width, height);

        for (size_t k = 0; k < limit; ++k) {
            if (k < height - 1) {
                U_reflectors.push_back(getLeftHouseholder(B, k));
                applyLeftHouseholder(B, U_reflectors.back(), k);
            }

            if (k < width - 2) { // Dopiero od width-2 ma sens dla prawego Householdera
                V1_reflectors.push_back(getRightHouseholder(B, k));
                applyRightHouseholder(B, V1_reflectors.back(), k);
            }
        }
        return { std::move(B), std::move(U_reflectors), std::move(V1_reflectors) };
    }

    [[nodiscard]] static BidiagonalForm<T> bidiagonalize(const Matrix<T>& m) {
        auto bEcon = bidiagonalizeEcon(m);

        return {
            buildU(bEcon.U_reflectors, m.getHeight()),
            std::move(bEcon.B),
            buildVT(bEcon.V1_reflectors, m.getWidth())
        };
    }

private:
    [[nodiscard]] static std::vector<T> getLeftHouseholder(const Matrix<T>& m, size_t k) {
        size_t height = m.getHeight();
        if (k >= height - 1) return {};

        std::vector<T> x(height - k);
        for (size_t i = k; i < height; ++i) {
            x[i - k] = m.getValue(k, i);
        }

        T xNorm = mfuncs::norm(x);
        if (xNorm > static_cast<T>(1e-15)) { // Zabezpieczenie przed wektorem zerowym
            T alpha = -mfuncs::signum(x[0]) * xNorm;
            x[0] -= alpha;
            mfuncs::normalise(x);
        }
        return x;
    }

    [[nodiscard]] static std::vector<T> getRightHouseholder(const Matrix<T>& m, size_t k) {
        size_t width = m.getWidth();
        if (k >= width - 1) return {};

        std::vector<T> x(width - k - 1);
        for (size_t i = k + 1; i < width; ++i) {
            x[i - (k + 1)] = m.getValue(i, k);
        }

        T xNorm = mfuncs::norm(x);
        if (xNorm > static_cast<T>(1e-15)) {
            T alpha = -mfuncs::signum(x[0]) * xNorm;
            x[0] -= alpha;
            mfuncs::normalise(x);
        }
        return x;
    }

    static void applyLeftHouseholder(Matrix<T>& m, const std::vector<T>& v, size_t k) {
        if (v.empty()) return;

        size_t height = m.getHeight(), width = m.getWidth();
        std::vector<T> w(width, T{});

        for (size_t x = k; x < width; ++x) { // Optymalizacja: omijamy już wyzerowane kolumny
            for (size_t y = k; y < height; ++y) {
                w[x] += v[y - k] * m.getValue(x, y);
            }
            w[x] *= static_cast<T>(2.0);
        }

        for (size_t y = k; y < height; ++y) {
            for (size_t x = k; x < width; ++x) { // Optymalizacja
                T val = m.getValue(x, y) - v[y - k] * w[x];
                m.setValue(x, y, val);
            }
        }
    }

    static void applyRightHouseholder(Matrix<T>& m, const std::vector<T>& vec, size_t k) {
        if (vec.empty()) return;

        size_t height = m.getHeight(), width = m.getWidth();
        std::vector<T> w(height, T{});

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = k + 1; x < width; ++x) {
                w[y] += m.getValue(x, y) * vec[x - (k + 1)];
            }
            w[y] *= static_cast<T>(2.0);
        }

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = k + 1; x < width; ++x) {
                T val = m.getValue(x, y) - w[y] * vec[x - (k + 1)];
                m.setValue(x, y, val);
            }
        }
    }

    [[nodiscard]] static Matrix<T> buildU(const std::vector<std::vector<T>>& U1, size_t n) {
        Matrix<T> U = Matrix<T>::identity(n);

        for (int k = static_cast<int>(U1.size()) - 1; k >= 0; --k) {
            const auto& vec = U1[k];
            if (vec.empty()) continue;

            for (size_t x = 0; x < n; ++x) {
                T dot = T{};
                for (size_t y = k; y < n; ++y) {
                    dot += vec[y - k] * U.getValue(x, y);
                }

                dot *= static_cast<T>(2.0);

                for (size_t y = k; y < n; ++y) {
                    T val = U.getValue(x, y) - vec[y - k] * dot;
                    U.setValue(x, y, val);
                }
            }
        }
        return U;
    }

    [[nodiscard]] static Matrix<T> buildVT(const std::vector<std::vector<T>>& V1, size_t n) {
        Matrix<T> VT = Matrix<T>::identity(n);

        for (int k = static_cast<int>(V1.size()) - 1; k >= 0; --k) {
            const auto& v = V1[k];
            if (v.empty()) continue;

            for (size_t y = 0; y < n; ++y) {
                T dot = T{};
                for (size_t x = static_cast<size_t>(k) + 1; x < n; ++x) {
                    dot += VT.getValue(x, y) * v[x - (k + 1)];
                }

                dot *= static_cast<T>(2.0);

                for (size_t x = static_cast<size_t>(k) + 1; x < n; ++x) {
                    T val = VT.getValue(x, y) - dot * v[x - (k + 1)];
                    VT.setValue(x, y, val);
                }
            }
        }
        return VT;
    }
};