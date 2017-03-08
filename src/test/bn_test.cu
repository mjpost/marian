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

#include "layers/generic.h"
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

  std::cerr << "Building graph" << std::endl;
  {
    auto graph = New<ExpressionGraph>();
    graph->setDevice(0);
    graph->reserveWorkspaceMB(128);

    auto x = graph->param("x", {batchSize, 1, 3, 3}, init=inits::from_vector(temp));

    auto filter = graph->param("filter", {6, 1, 3, 2}, init=inits::from_value(1.0f));

    auto y = convolution(x, filter);
    auto pool = max_pooling(y);

    auto cost = sum(pool, keywords::axis=1);

    debug(y, "y");
    debug(x, "x");
    debug(pool, "pool");
    debug(cost, "cost");
    debug(filter, "filter");

    std::cerr << "Forward" << std::endl;
    graph->forward();
    std::cerr << "Backward" << std::endl;
    graph->backward();

    std::cerr << "Forward" << std::endl;
    graph->forward();
    std::cerr << "Backward" << std::endl;
    graph->backward();

    std::vector<float> tmp(50);
    filter->grad() >> tmp;

  }

  return 0;
}
