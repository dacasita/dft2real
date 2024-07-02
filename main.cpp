#define _USE_MATH_DEFINES
#include <cmath>
#include "../matplotlibcpp.h"
#include "myfunctions.h"
#include <vector>
#include <complex>
#include <unordered_map>
#include <iostream>

namespace plt = matplotlibcpp;
#include <opencv2/opencv.hpp>

using namespace cv;

std::vector<std::complex<double>> dft(const std::vector<std::complex<double>>& z) {
  const int N = z.size();
  std::vector<std::complex<double>> Z(N);

  for (int k = 0; k < N; ++k) {
    Z[k] = std::complex<double>(0.0, 0.0);
    for (int n = 0; n < N; ++n) {
      double theta = 2.0 * M_PI * k * n / N;
      std::complex<double> wn(std::cos(theta), std::sin(theta));
      Z[k] += z[n] * wn;
    }
    // Normalize (optional for normalized DFT)
    // Z[k] /= N;
  }

  return Z;
}

std::vector<std::complex<double>> reverseAndConjugate(std::vector<std::complex<double>>& z) {
  const int N = z.size();
  std::vector<std::complex<double>> reversedZ(N);

  // Reverse order
  for (int i = 0; i < N; ++i) {
    reversedZ[N - i - 1] = z[i];
  }

  // Take conjugate
  for (auto& value : reversedZ) {
    value = std::conj(value);
  }

  return reversedZ;
}

int main() 
{
  // Define parameters
  const int N = 30;
  const double fs = 100;

  // Create a vector to store the signal values
  std::vector<double> x(N);

  // Generate cosine signal
  for (int i = 0; i < N; ++i) {
    double t = static_cast<double>(i) / fs;
    x[i] = std::cos(2.0 * M_PI * 130.0 * t);
  }

  // Create a vector to store the signal values
  std::vector<double> y(N);

  // Generate sine signal
  for (int i = 0; i < N; ++i) {
    double t = static_cast<double>(i) / fs;
    y[i] = std::sin(2.0 * M_PI * 180.0 * t);
  }

  // Create a vector of complex numbers
  std::vector<std::complex<double>> z(N);

  // Combine x and y into complex numbers
  for (int i = 0; i < N; ++i) {
    z[i] = std::complex<double>(x[i], y[i]);
  }

  // Perform DFT
  std::vector<std::complex<double>> Z = dft(z);

  // Reverse and conjugate Z
  std::vector<std::complex<double>> reversedZ = reverseAndConjugate(Z);

  std::vector<double> F_sample;
  std::vector<double> R_magnitudes;
  std::vector<std::complex<double>> X(N);
  for (int i = 0; i < N; ++i) {
    X[i] = (z[i] + reversedZ[i]) / 2.0;

    F_sample.push_back(i);
    R_magnitudes.push_back(std::abs(X[i])); 
  }

  std::vector<double> Y_magnitudes;
  std::vector<std::complex<double>> Y(N);
  for (int i = 0; i < N; ++i) 
  {
    Y[i] = (z[i] - reversedZ[i]) / (2.0 * std::complex<double>(0.0, 1.0));
    Y_magnitudes.push_back(std::abs(Y[i])); 
  }

  plt::plot(F_sample, R_magnitudes);
    plt::plot(F_sample, Y_magnitudes);
    plt::title("X(F) Y(F)");
    plt::xlabel("F?");
    plt::ylabel("X(F) Y(F)");
    plt::show();

  return 0;
}