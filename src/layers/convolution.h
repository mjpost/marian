#pragma once

#include <string>

#include "layers/generic.h"

namespace marian {

class Convolution : public Layer {
  public:
    Convolution(const std::string& name, int kernelHeight = 3, int kernelNum = 1, int depth = 1)
      : Layer(name), depth_(depth), kernelHeight_(kernelHeight), kernelNum_(kernelNum) {
    }

    Expr operator()(Expr x, Expr xMask) {
      auto graph = x->graph();
      const int batchDim = x->shape()[0];
      const int embDim = x->shape()[1];
      const int sentenceDim = x->shape()[2];

      std::vector<size_t> NCHWIndeces;
      for (int b = 0; b < batchDim; ++b) {
        for (int t = 0; t < sentenceDim; ++t) {
          NCHWIndeces.push_back((t * batchDim) + b);
        }
      }

      auto reshapedInput = reshape(x, {batchDim * sentenceDim, embDim, 1, x->shape()[3]});
      auto newX = reshape(rows(reshapedInput, NCHWIndeces), {batchDim, 1, sentenceDim, embDim});

      auto reshapedMask = reshape(xMask, {batchDim * sentenceDim, embDim, 1, x->shape()[3]});
      auto newMaskX = reshape(rows(reshapedMask, NCHWIndeces), {batchDim, 1, sentenceDim, embDim});

      Expr* previousInput = &newX;

      for (int layerIdx = 0; layerIdx < depth_; ++layerIdx) {
        auto kernels = graph->param(name_ + "kernel_" + std::to_string(layerIdx), {kernelNum_, kernelHeight_},
                                    keywords::init=inits::glorot_uniform);
        debug(kernels, "kernel_" + std::to_string(layerIdx));
        params_.push_back(kernels);
        auto input = *previousInput * newMaskX;
        auto output = tanh(input + convolution(input, kernels));
        debug(output, "PO" + std::to_string(layerIdx));
        previousInput = &output;
      }
      auto reshapedOutput = reshape(*previousInput, {batchDim * sentenceDim, x->shape()[1],
                                                     1, x->shape()[3]});

      std::vector<size_t> reverseNCHW;
      for (int t = 0; t < sentenceDim; ++t) {
        for (int b = 0; b < batchDim; ++b) {
          reverseNCHW.push_back(b * sentenceDim + t);
        }
      }

      auto reshaped = reshape(rows(reshapedOutput, reverseNCHW), x->shape());
      return reshaped * xMask;
    }

  protected:
    const int depth_;
    const int kernelHeight_;
    const int kernelNum_;
};

class Pooling : public Layer {
  public:
    Pooling(const std::string& name)
      : Layer(name) {
    }

    Expr operator()(Expr x, Expr xMask) {
      params_ = {};

      std::vector<size_t> newIndeces;
      int batchDim = x->shape()[0];
      int sentenceDim = x->shape()[2];

      for (int b = 0; b < batchDim; ++b) {
        for (int t = 0; t < sentenceDim; ++t) {
          newIndeces.push_back((t * batchDim) + b);
        }
      }
      // debug(x, "X");
      // debug(xMask, "xMask");

      auto masked = reshape(x * xMask, {batchDim * sentenceDim, x->shape()[1], 1, x->shape()[3]});
      auto newX = reshape(rows(masked, newIndeces), {batchDim, x->shape()[1], sentenceDim, 1});

      auto pooled = reshape(avg_pooling(newX), {batchDim * sentenceDim, x->shape()[1], 1, x->shape()[3]});

      // debug(masked, "masked");
      // debug(newX, "newX");
      // debug(pooled, "pooled");

      newIndeces.clear();
      for (int t = 0; t < sentenceDim; ++t) {
        for (int b = 0; b < batchDim; ++b) {
          newIndeces.push_back(b * sentenceDim + t);
        }
      }

      auto reshaped = reshape(rows(pooled, newIndeces), x->shape());
      return reshaped * xMask;
    }
};

}
