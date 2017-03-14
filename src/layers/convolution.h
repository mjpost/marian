#pragma once

#include <string>

#include "layers/generic.h"

namespace marian {

class Convolution : public Layer {
  public:
    Convolution(const std::string& name)
      : Layer(name) {
    }

    Expr operator()(Expr in, int kernelNum, int kernelHeight, int kernelWidth, int kernelHPad, int kernelWPad) {
      auto graph = in->graph();
      auto kernels = graph->param(name_ + "kernels", {kernelNum, 1, kernelHeight, kernelWidth},
          keywords::init=inits::from_value(1.0f));

      params_ = { kernels };

      auto conv = convolution(in, kernels, kernelHPad, kernelWPad);
      return conv;
    }
};


class MultiConvLayer : public Layer {
  public:
    MultiConvLayer(const std::string& name, int stackDim)
      : Layer(name),
        stackDim_(stackDim)
    {}


  Expr operator()(Expr x, Expr xMask) {
    auto masked = x * xMask;
    auto graph = x->graph();
    Expr* in = &masked;
    for (int idx = 0; idx < stackDim_; ++idx) {
      auto out = tanh(*in + Convolution(name_ + std::to_string(idx))(*in, 1, 3, 1, 1, 0));
      in = &out;
    }

    return *in;
  }

  protected:
    int stackDim_;
};

}
