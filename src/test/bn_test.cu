#include <iostream>
#include <cuda.h>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <functional>

#include "layers/convolution.h"
#include "marian.h"

int main(int argc, char** argv) {
  using namespace marian;
  using namespace data;
  using namespace keywords;

  auto options = New<Config>(argc, argv, false);

  int batchSize = 2;

  std::vector<float> temp(batchSize * 9);
  for (size_t i = 0; i < temp.size(); ++i) {
    temp[i] = i + 1;
  }

  const int numKernels = 6;
  const int kernelHeight = 3;
  const int kernelWidth = 2;

  std::cerr << "Building graph" << std::endl;
  {
    auto graph = New<ExpressionGraph>();
    graph->setDevice(0);
    graph->reserveWorkspaceMB(128);

    auto x = graph->param("x", {batchSize, 1, 3, 3}, init=inits::from_vector(temp));

    auto convLayer = Convolution("conv_layer")(x, numKernels, kernelHeight, kernelWidth, 0,0,
        -1, -1, 0, 0);

    /* auto y = convolution(x, filter); */
    /* auto pool = max_pooling(y); */

    auto cost = sum(convLayer, keywords::axis=1);

    debug(x, "x");
    debug(convLayer, "conv");
    debug(cost, "cost");

    std::cerr << "Forward" << std::endl;
    graph->forward();
    std::cerr << "Backward" << std::endl;
    graph->backward();
  }

  return 0;
}
