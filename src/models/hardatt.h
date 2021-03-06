#pragma once

#include "models/encdec.h"
#include "models/s2s.h"
#include "models/multi_s2s.h"

namespace marian {

class DecoderStateHardAtt : public DecoderState {
  protected:
    std::vector<Expr> states_;
    Expr probs_;
    Ptr<EncoderState> encState_;
    std::vector<size_t> attentionIndices_;
    
  public:
    DecoderStateHardAtt(const std::vector<Expr> states,
                     Expr probs,
                     Ptr<EncoderState> encState,
                     const std::vector<size_t>& attentionIndices)
    : states_(states), probs_(probs), encState_(encState),
      attentionIndices_(attentionIndices) {}
    
    
    virtual Ptr<EncoderState> getEncoderState() { return encState_; }
    virtual Expr getProbs() { return probs_; }
    virtual void setProbs(Expr probs) { probs_ = probs; }
    
    virtual Ptr<DecoderState> select(const std::vector<size_t>& selIdx) {
      int numSelected = selIdx.size();
      int dimState = states_[0]->shape()[1];
      
      std::vector<Expr> selectedStates;
      for(auto state : states_) {
        selectedStates.push_back(
          reshape(rows(state, selIdx),
                  {1, dimState, 1, numSelected})
        );
      }
      
      std::vector<size_t> selectedAttentionIndices;
      for(auto i : selIdx)
        selectedAttentionIndices.push_back(attentionIndices_[i]);
      
      return New<DecoderStateHardAtt>(selectedStates, probs_, encState_,
                                      selectedAttentionIndices);
    }

    virtual void setAttentionIndices(const std::vector<size_t>& attentionIndices) {
      attentionIndices_ = attentionIndices;
    }
    
    virtual std::vector<size_t>& getAttentionIndices() {
      UTIL_THROW_IF2(attentionIndices_.empty(), "Empty attention indices");
      return attentionIndices_;
    }
    
    virtual const std::vector<Expr>& getStates() { return states_; }
    
    //virtual const std::vector<float> breakDown(size_t i) {
    //  auto costs = DecoderState::breakDown(i);
    //  costs.resize(5, 0);
    //  
    //  int vocabSize = getProbs()->shape()[1];
    //  int e = i % vocabSize;
    //  int h = i / vocabSize;
    //  
    //  int a = attentionIndices_[h];
    //  
    //  auto& words = getEncoderState()->getSourceWords();
    //  
    //  if(e != 2) {
    //    costs[1] = 1;
    //    
    //    if(words[a] == e)
    //      costs[2] = 1;
    //    else
    //      costs[3] = 1;
    //      
    //    costs[4] = std::find(words.begin(), words.end(), e) == words.end();
    //  }
    //  
    //  return costs;
    //}
};

class DecoderHardAtt : public DecoderBase {
  protected:
    Ptr<RNN<GRU>> rnnL1;
    Ptr<MLRNN<GRU>> rnnLn;
    std::unordered_set<Word> specialSymbols_;
  
  public:

    template <class ...Args>
    DecoderHardAtt(Ptr<Config> options, Args ...args)
     : DecoderBase(options, args...) {
    
      if(options->has("special-vocab")) {
        auto spec = options->get<std::vector<size_t>>("special-vocab");
        specialSymbols_.insert(spec.begin(), spec.end());
      }
    }

    virtual Ptr<DecoderState> startState(Ptr<EncoderState> encState) {
      using namespace keywords;

      auto meanContext = weighted_average(encState->getContext(),
                                          encState->getMask(),
                                          axis=2);

      bool layerNorm = options_->get<bool>("layer-normalization");
      auto start = Dense("ff_state",
                         options_->get<int>("dim-rnn"),
                         activation=act::tanh,
                         normalize=layerNorm)(meanContext);
      
      std::vector<Expr> startStates(options_->get<size_t>("layers-dec"), start);
      return New<DecoderStateHardAtt>(startStates, nullptr, encState,
                                      std::vector<size_t>({0}));
    }
     
    virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                   Ptr<DecoderState> state) {
      using namespace keywords;

      int dimTrgVoc = options_->get<std::vector<int>>("dim-vocabs").back();
      
      int dimTrgEmb = options_->get<int>("dim-emb")
                    + options_->get<int>("dim-pos");
      
                    
      int dimDecState = options_->get<int>("dim-rnn");
      bool layerNorm = options_->get<bool>("layer-normalization");
      bool skipDepth = options_->get<bool>("skip");
      size_t decoderLayers = options_->get<size_t>("layers-dec");

      float dropoutRnn = inference_ ? 0 : options_->get<float>("dropout-rnn");
      float dropoutTrg = inference_ ? 0 : options_->get<float>("dropout-trg");

      auto stateHardAtt = std::dynamic_pointer_cast<DecoderStateHardAtt>(state);
      
      auto trgEmbeddings = stateHardAtt->getTargetEmbeddings();
      
      auto context = stateHardAtt->getEncoderState()->getContext();
      int dimContext = context->shape()[1];
      int dimSrcWords = context->shape()[2];
      
      int dimBatch = context->shape()[0];
      int dimTrgWords = trgEmbeddings->shape()[2];
      int dimBeam = trgEmbeddings->shape()[3];
            
      if(dropoutTrg) {
        auto trgWordDrop = graph->dropout(dropoutTrg, {dimBatch, 1, dimTrgWords});
        trgEmbeddings = dropout(trgEmbeddings, mask=trgWordDrop);
      }
      
      auto flatContext = reshape(context, {dimBatch * dimSrcWords, dimContext});
      auto attendedContext = rows(flatContext, stateHardAtt->getAttentionIndices());
      attendedContext = reshape(attendedContext, {dimBatch, dimContext, dimTrgWords, dimBeam});
      
      auto rnnInputs = concatenate({trgEmbeddings, attendedContext}, axis=1);
      int dimInput = rnnInputs->shape()[1];
    
      if(!rnnL1)
        rnnL1 = New<RNN<GRU>>(graph, "decoder",
                              dimInput, dimDecState,
                              dropout_prob=dropoutRnn,
                              normalize=layerNorm);

      auto stateL1 = (*rnnL1)(rnnInputs, stateHardAtt->getStates()[0]);
      
      std::vector<Expr> statesOut;
      statesOut.push_back(stateL1);

      Expr outputLn;
      if(decoderLayers > 1) {
        std::vector<Expr> statesIn;
        for(int i = 1; i < stateHardAtt->getStates().size(); ++i)
          statesIn.push_back(stateHardAtt->getStates()[i]);

        if(!rnnLn) 
          rnnLn = New<MLRNN<GRU>>(graph, "decoder",
                                  decoderLayers - 1,
                                  dimDecState, dimDecState,
                                  normalize=layerNorm,
                                  dropout_prob=dropoutRnn,
                                  skip=skipDepth,
                                  skip_first=skipDepth);
        
        std::vector<Expr> statesLn;
        std::tie(outputLn, statesLn) = (*rnnLn)(stateL1, statesIn);

        statesOut.insert(statesOut.end(),
                         statesLn.begin(), statesLn.end());
      }
      else {
        outputLn = stateL1;
      }

      //// 2-layer feedforward network for outputs and cost
      auto logitsL1 = Dense("ff_logit_l1", dimTrgEmb,
                            activation=act::tanh,
                            normalize=layerNorm)
                        (rnnInputs, outputLn);

      auto logitsOut = Dense("ff_logit_l2", dimTrgVoc)(logitsL1);    
      
      return New<DecoderStateHardAtt>(statesOut, logitsOut,
                                      stateHardAtt->getEncoderState(),
                                      stateHardAtt->getAttentionIndices());
    }
    
    
    virtual std::tuple<Expr, Expr>
    groundTruth(Ptr<DecoderState> state,
                Ptr<ExpressionGraph> graph,
                Ptr<data::CorpusBatch> batch,
                size_t index) {
      using namespace keywords;

      auto ret = DecoderBase::groundTruth(state, graph, batch, index);
      
      auto subBatch = (*batch)[index];
      int dimBatch = subBatch->batchSize();
      int dimWords = subBatch->batchWidth();
      
      std::vector<size_t> attentionIndices(dimBatch, 0);
      std::vector<size_t> currentPos(dimBatch, 0);
      std::iota(currentPos.begin(), currentPos.end(), 0);

      for(int i = 0; i < dimWords - 1; ++i) {
        for(int j = 0; j < dimBatch; ++j) {
          size_t word = subBatch->indeces()[i * dimBatch + j];
          if(specialSymbols_.count(word))
            currentPos[j] += dimBatch;
          attentionIndices.push_back(currentPos[j]);
        }
      }
      
      std::dynamic_pointer_cast<DecoderStateHardAtt>(state)->setAttentionIndices(attentionIndices);
            
      return ret;
    }
    
    virtual void selectEmbeddings(Ptr<ExpressionGraph> graph,
                                  Ptr<DecoderState> state,
                                  const std::vector<size_t>& embIdx) {
      DecoderBase::selectEmbeddings(graph, state, embIdx);
      
      auto stateHardAtt = std::dynamic_pointer_cast<DecoderStateHardAtt>(state);
      
      int dimSrcWords = state->getEncoderState()->getContext()->shape()[2];

      if(embIdx.empty()) {
        stateHardAtt->setAttentionIndices({0});  
      }
      else {
        for(size_t i = 0; i < embIdx.size(); ++i)
          if(specialSymbols_.count(embIdx[i])) {
            stateHardAtt->getAttentionIndices()[i]++;
            if(stateHardAtt->getAttentionIndices()[i] >= dimSrcWords)
              stateHardAtt->getAttentionIndices()[i] = dimSrcWords - 1;
          }
      }
    }

};

typedef EncoderDecoder<EncoderS2S, DecoderHardAtt> HardAtt;


/******************************************************************************/


typedef AttentionCell<GRU, GlobalAttention, GRU> CGRU;

class DecoderHardSoftAtt : public DecoderHardAtt {
  private:
    Ptr<GlobalAttention> attention_;
    Ptr<RNN<CGRU>> rnnL1;
    Ptr<MLRNN<GRU>> rnnLn;
    
  public:
    template <class ...Args>
    DecoderHardSoftAtt(Ptr<Config> options, Args ...args)
     : DecoderHardAtt(options, args...) {}
    
    virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                   Ptr<DecoderState> state) {
      using namespace keywords;

      int dimTrgVoc = options_->get<std::vector<int>>("dim-vocabs").back();
      
      int dimTrgEmb = options_->get<int>("dim-emb")
                    + options_->get<int>("dim-pos");
      
                    
      int dimDecState = options_->get<int>("dim-rnn");
      bool layerNorm = options_->get<bool>("layer-normalization");
      bool skipDepth = options_->get<bool>("skip");
      size_t decoderLayers = options_->get<size_t>("layers-dec");

      float dropoutRnn = inference_ ? 0 : options_->get<float>("dropout-rnn");
      float dropoutTrg = inference_ ? 0 : options_->get<float>("dropout-trg");

      auto stateHardAtt = std::dynamic_pointer_cast<DecoderStateHardAtt>(state);
      
      auto trgEmbeddings = stateHardAtt->getTargetEmbeddings();
      
      auto context = stateHardAtt->getEncoderState()->getContext();
      int dimContext = context->shape()[1];
      int dimSrcWords = context->shape()[2];
      
      int dimBatch = context->shape()[0];
      int dimTrgWords = trgEmbeddings->shape()[2];
      int dimBeam = trgEmbeddings->shape()[3];
            
      if(dropoutTrg) {
        auto trgWordDrop = graph->dropout(dropoutTrg, {dimBatch, 1, dimTrgWords});
        trgEmbeddings = dropout(trgEmbeddings, mask=trgWordDrop);
      }
      

      auto flatContext = reshape(context, {dimBatch * dimSrcWords, dimContext});
      auto attendedContext = rows(flatContext, stateHardAtt->getAttentionIndices());
      attendedContext = reshape(attendedContext, {dimBatch, dimContext, dimTrgWords, dimBeam});
      
      auto rnnInputs = concatenate({trgEmbeddings, attendedContext}, axis=1);
      int dimInput = rnnInputs->shape()[1];
      
      if(!attention_)
        attention_ = New<GlobalAttention>("decoder",
                                          state->getEncoderState(),
                                          dimDecState,
                                          dropout_prob=dropoutRnn,
                                          normalize=layerNorm);
      
      if(!rnnL1)
        rnnL1 = New<RNN<CGRU>>(graph, "decoder",
                               dimInput, dimDecState,
                               attention_,
                               dropout_prob=dropoutRnn,
                               normalize=layerNorm);

      auto stateL1 = (*rnnL1)(rnnInputs, stateHardAtt->getStates()[0]);
      
      bool single = stateHardAtt->doSingleStep();  
      auto alignedContext = single ?
        rnnL1->getCell()->getLastContext() :
        rnnL1->getCell()->getContexts();
      
      std::vector<Expr> statesOut;
      statesOut.push_back(stateL1);

      Expr outputLn;
      if(decoderLayers > 1) {
        std::vector<Expr> statesIn;
        for(int i = 1; i < stateHardAtt->getStates().size(); ++i)
          statesIn.push_back(stateHardAtt->getStates()[i]);

        if(!rnnLn) 
          rnnLn = New<MLRNN<GRU>>(graph, "decoder",
                                  decoderLayers - 1,
                                  dimDecState, dimDecState,
                                  normalize=layerNorm,
                                  dropout_prob=dropoutRnn,
                                  skip=skipDepth,
                                  skip_first=skipDepth);
        
        std::vector<Expr> statesLn;
        std::tie(outputLn, statesLn) = (*rnnLn)(stateL1, statesIn);

        statesOut.insert(statesOut.end(),
                         statesLn.begin(), statesLn.end());
      }
      else {
        outputLn = stateL1;
      }

      //// 2-layer feedforward network for outputs and cost
      auto logitsL1 = Dense("ff_logit_l1", dimTrgEmb,
                            activation=act::tanh,
                            normalize=layerNorm)
                        (rnnInputs, outputLn, alignedContext);

      auto logitsOut = Dense("ff_logit_l2", dimTrgVoc)(logitsL1);    
      
      return New<DecoderStateHardAtt>(statesOut, logitsOut,
                                      stateHardAtt->getEncoderState(),
                                      stateHardAtt->getAttentionIndices());
    }
};

typedef EncoderDecoder<EncoderS2S, DecoderHardSoftAtt> HardSoftAtt;

class MultiDecoderHardSoftAtt : public DecoderHardSoftAtt {
  private:
    Ptr<GlobalAttention> attention1_;
    Ptr<GlobalAttention> attention2_;
    Ptr<RNN<MultiCGRU>> rnnL1;
    Ptr<MLRNN<GRU>> rnnLn;
    
  public:
    template <class ...Args>
    MultiDecoderHardSoftAtt(Ptr<Config> options, Args ...args)
     : DecoderHardSoftAtt(options, args...) {}
    
     virtual Ptr<DecoderState> startState(Ptr<EncoderState> encState) {
      using namespace keywords;

      auto mEncState = std::static_pointer_cast<EncoderStateMultiS2S>(encState);

      auto meanContext1 = weighted_average(mEncState->enc1->getContext(),
                                           mEncState->enc1->getMask(),
                                           axis=2);

      auto meanContext2 = weighted_average(mEncState->enc2->getContext(),
                                           mEncState->enc2->getMask(),
                                           axis=2);

      bool layerNorm = options_->get<bool>("layer-normalization");

      auto start = Dense("ff_state",
                         options_->get<int>("dim-rnn"),
                         activation=act::tanh,
                         normalize=layerNorm)(meanContext1, meanContext2);
      
      std::vector<Expr> startStates(options_->get<size_t>("layers-dec"), start);
      return New<DecoderStateHardAtt>(startStates, nullptr, encState,
                                      std::vector<size_t>({0}));
    }
    
    virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                   Ptr<DecoderState> state) {
      using namespace keywords;

      int dimTrgVoc = options_->get<std::vector<int>>("dim-vocabs").back();
      
      int dimTrgEmb = options_->get<int>("dim-emb")
                    + options_->get<int>("dim-pos");
      
                    
      int dimDecState = options_->get<int>("dim-rnn");
      bool layerNorm = options_->get<bool>("layer-normalization");
      bool skipDepth = options_->get<bool>("skip");
      size_t decoderLayers = options_->get<size_t>("layers-dec");

      float dropoutRnn = inference_ ? 0 : options_->get<float>("dropout-rnn");
      float dropoutTrg = inference_ ? 0 : options_->get<float>("dropout-trg");

      auto mEncState
        = std::static_pointer_cast<EncoderStateMultiS2S>(state->getEncoderState());
      
      auto stateHardAtt = std::dynamic_pointer_cast<DecoderStateHardAtt>(state);
      
      auto trgEmbeddings = stateHardAtt->getTargetEmbeddings();
      
      auto context = mEncState->enc1->getContext();
      int dimContext = context->shape()[1];
      int dimSrcWords = context->shape()[2];
      
      int dimBatch = context->shape()[0];
      int dimTrgWords = trgEmbeddings->shape()[2];
      int dimBeam = trgEmbeddings->shape()[3];
            
      if(dropoutTrg) {
        auto trgWordDrop = graph->dropout(dropoutTrg, {dimBatch, 1, dimTrgWords});
        trgEmbeddings = dropout(trgEmbeddings, mask=trgWordDrop);
      }

      auto flatContext = reshape(context, {dimBatch * dimSrcWords, dimContext});
      auto attendedContext = rows(flatContext, stateHardAtt->getAttentionIndices());
      attendedContext = reshape(attendedContext, {dimBatch, dimContext, dimTrgWords, dimBeam});
      
      auto rnnInputs = concatenate({trgEmbeddings, attendedContext}, axis=1);
      int dimInput = rnnInputs->shape()[1];
      
      if(!attention1_)
        attention1_ = New<GlobalAttention>("decoder_att1",
                                           mEncState->enc1,
                                           dimDecState,
                                           normalize=layerNorm);
      if(!attention2_)
        attention2_ = New<GlobalAttention>("decoder_att2",
                                           mEncState->enc2,
                                           dimDecState,
                                           normalize=layerNorm);
      
      if(!rnnL1)
        rnnL1 = New<RNN<MultiCGRU>>(graph, "decoder",
                                    dimInput, dimDecState,
                                    attention1_, attention2_,
                                    dropout_prob=dropoutRnn,
                                    normalize=layerNorm);

      auto stateL1 = (*rnnL1)(rnnInputs, stateHardAtt->getStates()[0]);
      
      bool single = stateHardAtt->doSingleStep();  
      auto alignedContext = single ?
        rnnL1->getCell()->getLastContext() :
        rnnL1->getCell()->getContexts();
      
      std::vector<Expr> statesOut;
      statesOut.push_back(stateL1);

      Expr outputLn;
      if(decoderLayers > 1) {
        std::vector<Expr> statesIn;
        for(int i = 1; i < stateHardAtt->getStates().size(); ++i)
          statesIn.push_back(stateHardAtt->getStates()[i]);

        if(!rnnLn) 
          rnnLn = New<MLRNN<GRU>>(graph, "decoder",
                                  decoderLayers - 1,
                                  dimDecState, dimDecState,
                                  normalize=layerNorm,
                                  dropout_prob=dropoutRnn,
                                  skip=skipDepth,
                                  skip_first=skipDepth);
        
        std::vector<Expr> statesLn;
        std::tie(outputLn, statesLn) = (*rnnLn)(stateL1, statesIn);

        statesOut.insert(statesOut.end(),
                         statesLn.begin(), statesLn.end());
      }
      else {
        outputLn = stateL1;
      }

      //// 2-layer feedforward network for outputs and cost
      auto logitsL1 = Dense("ff_logit_l1", dimTrgEmb,
                            activation=act::tanh,
                            normalize=layerNorm)
                        (rnnInputs, outputLn, alignedContext);

      auto logitsOut = Dense("ff_logit_l2", dimTrgVoc)(logitsL1);    
      
      return New<DecoderStateHardAtt>(statesOut, logitsOut,
                                      stateHardAtt->getEncoderState(),
                                      stateHardAtt->getAttentionIndices());
    }
    
    virtual void selectEmbeddings(Ptr<ExpressionGraph> graph,
                                  Ptr<DecoderState> state,
                                  const std::vector<size_t>& embIdx) {
      DecoderBase::selectEmbeddings(graph, state, embIdx);
      
      auto stateHardAtt = std::dynamic_pointer_cast<DecoderStateHardAtt>(state);
      
       auto mEncState
        = std::static_pointer_cast<EncoderStateMultiS2S>(state->getEncoderState());
      
      int dimSrcWords = mEncState->enc1->getContext()->shape()[2];

      if(embIdx.empty()) {
        stateHardAtt->setAttentionIndices({0});  
      }
      else {
        for(size_t i = 0; i < embIdx.size(); ++i)
          if(specialSymbols_.count(embIdx[i])) {
            stateHardAtt->getAttentionIndices()[i]++;
            if(stateHardAtt->getAttentionIndices()[i] >= dimSrcWords)
              stateHardAtt->getAttentionIndices()[i] = dimSrcWords - 1;
          }
      }
    }
};

class MultiHardSoftAtt : public EncoderDecoder<MultiEncoderS2S, MultiDecoderHardSoftAtt> {
  public:
    template <class ...Args>
    MultiHardSoftAtt(Ptr<Config> options, Args ...args)
    : EncoderDecoder(options, {0, 1, 2}, args...) {}
};


}
