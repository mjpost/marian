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

  int batchSize = 1;

  std::vector<float> temp(batchSize * 9);
  for (size_t i = 0; i < temp.size(); ++i) {
    temp[i] = i + 1;
  }
  std::vector<float> temp2(3072 * 3072);
  std::vector<float> indeces(batchSize, 0.f);


  std::cerr << "Building graph" << std::endl;
  {
    auto graph = New<ExpressionGraph>();
    graph->setDevice(0);
    graph->reserveWorkspaceMB(128);

    std::cerr << "Setting X" << std::endl;
    auto x = graph->param("x", {batchSize, 1, 3, 3}, init=inits::from_vector(temp));

    std::cerr << "Setting filter" << std::endl;
    auto filter = graph->param("gamma", {16, 1, 3, 2}, init=inits::from_value(1.0f));

    std::cerr << "Setting convolution" << std::endl;
    auto y = convolution(x, filter);
    auto pool = max_pooling(y);

    auto idx = graph->constant(shape={16, 1},
                               init=inits::from_value(1.0f));
    auto ce = cross_entropy(pool, idx);
    auto cost = mean(sum(ce, keywords::axis=1), keywords::axis=1);

    debug(y, "y");
    debug(x, "x");
    debug(ce, "ce");
    debug(pool, "pool");
    debug(cost, "cost");
    debug(filter, "filter");

    std::cerr << "Forward" << std::endl;
    graph->forward();
    std::cerr << "Backward" << std::endl;
    graph->backward();
  }

  return 0;
}
