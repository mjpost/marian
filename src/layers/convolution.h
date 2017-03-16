#pragma once

#include <string>

#include "layers/generic.h"

namespace marian {

class Convolution : public Layer {
  public:
    Convolution(const std::string& name)
      : Layer(name) {
    }

    Expr operator()(Expr in) {
      auto graph = in->graph();
      auto& inShape = in->shape();
      auto kernels = graph->param(name_ + "filter_", {inShape[1], inShape[1], 3},
                                  keywords::init=inits::glorot_uniform);

      params_ = { kernels };

      auto conv = convolution(in, kernels);
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
      // auto out = tanh(*in + Convolution(name_ + std::to_string(idx))(*in));
      auto out = Convolution(name_ + std::to_string(idx))(*in);
       in = &out;
    }

    return *in;
  }

  protected:
    int stackDim_;
};

}
