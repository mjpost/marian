#pragma once

#include <string>

#include "layers/generic.h"

namespace marian {


class Convolution : public Layer {
  public:
    Convolution(const std::string& name)
      : Layer(name) {
    }

    Expr operator()(Expr in, int kernelNum, int kernelHeight, int kernelWidth, int kernelHPad, int kernelWPad,
        int poolingHeight, int poolingWidth, int poolingHPad, int poolingWPad) {
      auto graph = in->graph();
      auto kernels = graph->param(name_ + "kernels", {kernelNum, 1, kernelHeight, kernelWidth},
          keywords::init=inits::from_value(1.0f));
          // keywords::init=inits::glorot_uniform);

      params_ = { kernels };

      auto conv = convolution(in, kernels, kernelHPad, kernelWPad);
      auto pooling = max_pooling(conv, poolingHPad, poolingWPad, poolingHeight, poolingWidth);
      return pooling;
    }
};

}
