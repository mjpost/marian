#pragma once


#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "shape.h"
#include "common/logging.h"

#define THREAD_GUARD(body) std::thread([&]() { body ; }).join()
#define NodeOp(op) [=]() { op ; }

namespace marian {

  template<class T>
  using Ptr = std::shared_ptr<T>;

  template<class T>
  using UPtr = std::unique_ptr<T>;
  
  template<class T>
  using Weak = std::weak_ptr<T>;

  /** @brief Creates shared_ptr of any type, passes all arguments to any available constructor */
  template <class T, typename ...Args>
  Ptr<T> New(Args&& ... args) {
    return Ptr<T>(new T(std::forward<Args>(args)...));
  }

  template <class T>
  Ptr<T> New(Ptr<T> p) {
    return Ptr<T>(p);
  }

  typedef float Float;

  template<class T>
  using DeviceVector = thrust::device_vector<T>;

  template<class T>
  using HostVector = thrust::host_vector<T>;
}


#include "keywords.h"

namespace marian {

  class TensorBase;
  typedef Ptr<TensorBase> Tensor;

  template <class DataType> class Chainable;
  typedef Ptr<Chainable<Tensor>> Expr;

  class OptimizerBase;
  typedef Ptr<OptimizerBase> OptimizerBasePtr;

  class ClipperBase;
  typedef Ptr<ClipperBase> ClipperBasePtr;

  class RunBase;
  typedef Ptr<RunBase> RunBasePtr;

  class LexProbs;
  
  // An enumeration of activations
  enum struct act { linear, tanh, logit, ReLU };

  // An enumeration of directions
  enum struct dir { forward, backward, bidirect };

  /**
   * @brief Defines a set of keywords.
   *
   * Each invocation of the KEY(name, value_type) macro
   *    will result in the creation of an instance of the Keyword class.
   */
  namespace keywords {
    KEY(axis, int);
    KEY(shape, Shape);
    KEY(value, float);
    KEY(prefix, std::string);
    KEY(final, bool);
    KEY(output_last, bool);
    KEY(activation, act);
    KEY(direction, dir);
    KEY(mask, Expr);
    KEY(dropout_prob, float);
    KEY(init, std::function<void(Tensor)>);


    KEY(eta, float);
    KEY(beta1, float);
    KEY(beta2, float);
    KEY(eps, float);
    KEY(optimizer, Ptr<OptimizerBase>);
    KEY(clip, Ptr<ClipperBase>);
    KEY(batch_size, int);
    KEY(normalize, bool);
    KEY(inference, bool);
    KEY(skip, bool);
    KEY(skip_first, bool);
    KEY(coverage, Expr);
    KEY(max_epochs, int);
    KEY(valid, Ptr<RunBase>);
    KEY(lex_probs, Ptr<LexProbs>);
  }

}
