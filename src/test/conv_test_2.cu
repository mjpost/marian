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

  int numWords = 4;
  int dimWord = 3;
  int numBatches = 2;
  int numLayers = 1;

  std::vector<float> embData(numBatches * numWords * dimWord);
  std::vector<float> embMask(numBatches * numWords * dimWord);

  for (size_t i = 0; i < embData.size(); ++i) {
    embData[i] = i;
    embMask[i] = 1;
  }

  std::cerr << "Building graph" << std::endl;
  {
    auto graph = New<ExpressionGraph>();
    graph->setDevice(0);
    graph->reserveWorkspaceMB(128);

    auto x = graph->param("x", {numBatches, dimWord, numWords}, init=inits::from_vector(embData));

    auto xMask = graph->constant(shape={numBatches, 1, numWords}, init=inits::from_vector(embMask));

    auto convLayer = MultiConvLayer("enc_", 2) (x, xMask);

    auto cost = sum(convLayer, keywords::axis=1);

    debug(x, "x");
    debug(convLayer, "conv");
    /* debug(cost, "cost"); */

    std::cerr << "Forward" << std::endl;
    graph->forward();
    cudaDeviceSynchronize();
    std::vector<float> tmp;
    convLayer->val() >> tmp;
    for (auto f :  tmp) std::cerr << f  << " ";
    std::cerr << std::endl;
    std::cerr << "Backward" << std::endl;
    graph->backward();
    tmp.clear();
    convLayer->grad() >> tmp;
    cudaDeviceSynchronize();
    std::cerr << "kon\n";
    for (auto f :  tmp) std::cerr << f  << " ";
    std::cerr << std::endl;
  }

  return 0;
}
