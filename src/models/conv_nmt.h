#pragma once

#include "models/s2s.h"
#include "layers/attention.h"
#include "layers/convolution.h"
#include "common/logging.h"

namespace marian {

class ConvEncoderState : public EncoderStateS2S {
  public:
    ConvEncoderState(Expr aContext, Expr cContext, Expr mask);
    Expr getConvContext();

  private:
    Expr convContext_;
};


class PoolingEncoder : public EncoderBase {
  public:
    template <class ...Args>
    PoolingEncoder(Ptr<Config> options, Args ...args)
     : EncoderBase(options, args...)
    {}

    Ptr<EncoderState>
    build(Ptr<ExpressionGraph> graph,
          Ptr<data::CorpusBatch> batch, size_t batchIdx = 0);

  protected:
    std::tuple<Expr, Expr>
    prepareSource(Expr emb, Expr posEmb, Ptr<data::CorpusBatch> batch, size_t index);
};


class ConvolutionalEncoder : public EncoderBase {
  public:
    template <class ...Args>
    ConvolutionalEncoder(Ptr<Config> options, Args ...args)
     : EncoderBase(options, args...)
    {}

    Ptr<EncoderState>
    build(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, size_t batchIdx= 0);

  protected:
    std::tuple<Expr, Expr> prepareSource(Expr emb, Ptr<data::CorpusBatch> batch, size_t index);
};


class ConvolutionalDecoder : public DecoderBase {
  private:
    Ptr<GlobalAttention> attention_;

  public:
    template <class ...Args>
    ConvolutionalDecoder(Ptr<Config> options, Args ...args)
     : DecoderBase(options, args...)
    {}

    Ptr<DecoderState> startState(Ptr<EncoderState> encState);

    virtual Ptr<DecoderState> step(Expr embeddings,
                                   Ptr<DecoderState>,
                                   bool single=false) ;

};


class ConvNMT : public EncoderDecoder<PoolingEncoder, ConvolutionalDecoder> {
  public:
    template <class ...Args>
    ConvNMT(Ptr<Config> options, Args ...args)
      : EncoderDecoder(options, args...) {
    }
};

}  // namespace marian
