#include "models/conv_nmt.h"
#include "models/encdec.h"


namespace marian {


ConvEncoderState::ConvEncoderState(Expr aContext, Expr cContext, Expr mask)
  : EncoderStateS2S(aContext, mask),
    convContext_(cContext)
{}


Expr ConvEncoderState::getConvContext() {
  return convContext_;
}


Ptr<EncoderState>
PoolingEncoder::build(Ptr<ExpressionGraph> graph,
                      Ptr<data::CorpusBatch> batch,
                      size_t batchIdx) {
  using namespace keywords;

  int dimSrcVoc = options_->get<std::vector<int>>("dim-vocabs")[batchIdx];
  int dimSrcEmb = options_->get<int>("dim-emb");
  int maxSrcLength = options_->get<int>("max-length");

  float dropoutSrc = inference_ ? 0 : options_->get<float>("dropout-src");

  auto xEmb = Embedding("Wemb", dimSrcVoc, dimSrcEmb)(graph);
  auto posEmb = Embedding("Wemb_pos", maxSrcLength, dimSrcEmb)(graph);

  Expr x, xMask;

  std::tie(x, xMask) = prepareSource(xEmb, posEmb, batch, batchIdx);

  std::string convType = options_->get<std::string>("conv-enc-type");

  if (dropoutSrc) {
    int srcWords = x->shape()[2];
    auto srcWordDrop = graph->dropout(dropoutSrc, {1, 1, srcWords});
    x = dropout(x, mask=srcWordDrop);
  }

  Expr attContext;
  Expr srcContext;
  if (convType == "pooling") {
    attContext = Pooling("enc_pooling")(x, xMask);
    srcContext = x;
  } else if (convType == "avg") {
    attContext = Pooling("enc_pooling")(x, xMask);
    srcContext = x;
  } else if (convType == "full") {
    attContext = Pooling("enc_pooling")(x, xMask);
    srcContext = x;
  }
  /* debug(attContext, "ATT"); */
  /* debug(srcContext, "SRC"); */


  return New<ConvEncoderState>(attContext, srcContext, xMask);
}


std::tuple<Expr, Expr>
PoolingEncoder::prepareSource(Expr emb, Expr posEmb, Ptr<data::CorpusBatch> batch, size_t index) {
  using namespace keywords;


  auto& wordIndeces = batch->at(index)->indeces();

  auto& mask = batch->at(index)->mask();

  std::vector<size_t> posIndeces;

  for (size_t iPos = 0; iPos < batch->at(index)->batchWidth(); ++iPos) {
    for (size_t i = 0; i < batch->at(index)->batchSize(); ++i) {
      if (iPos < posEmb->shape()[0]) {
        posIndeces.push_back(iPos);
      } else {
        posIndeces.push_back(posEmb->shape()[0] - 1);
      }
    }
  }

  int batchSize = batch->size();
  int dimEmb = emb->shape()[1];
  int batchLength = batch->at(index)->batchWidth();

  auto graph = emb->graph();

  auto xWord = reshape(rows(emb, wordIndeces), {batchSize, dimEmb, batchLength});
  auto xPos = reshape(rows(posEmb, posIndeces), {batchSize, dimEmb, batchLength});
  auto x = xWord + xPos;
  auto xMask = graph->constant(shape={batchSize, 1, batchLength}, init=inits::from_vector(mask));
  return std::make_tuple(x, xMask);
}


Ptr<EncoderState>
ConvolutionalEncoder::build(Ptr<ExpressionGraph> graph,
                            Ptr<data::CorpusBatch> batch,
                            size_t batchIdx) {
  using namespace keywords;

  int dimSrcVoc = options_->get<std::vector<int>>("dim-vocabs")[batchIdx];
  int dimSrcEmb = options_->get<int>("dim-emb");
  int maxSrcLength = options_->get<int>("max-length");
  dimSrcVoc += maxSrcLength;

  // bool layerNorm = options_->get<bool>("normalize");

  float dropoutSrc = inference_ ? 0 : options_->get<float>("dropout-src");

  auto xEmb = Embedding("Wemb", dimSrcVoc, dimSrcEmb)(graph);

  Expr x, xMask;
  std::tie(x, xMask) = prepareSource(xEmb, batch, batchIdx);

  if(dropoutSrc) {
    int srcWords = x->shape()[2];
    auto srcWordDrop = graph->dropout(dropoutSrc, {1, 1, srcWords});
    x = dropout(x, mask=srcWordDrop);
  }

  int stackDim = 3;
  auto aContext = Convolution("enc_cnn-a_", 3, 1, 2 * stackDim)(x, xMask);
  auto cContext = Convolution("enc_cnn-c_", 3, 1, stackDim)(x, xMask);

  return New<ConvEncoderState>(aContext, cContext, xMask);
}

std::tuple<Expr, Expr>
ConvolutionalEncoder::prepareSource(Expr emb, Ptr<data::CorpusBatch> batch, size_t index) {
  using namespace keywords;

  auto& wordIndeces = batch->at(index)->indeces();
  auto& mask = batch->at(index)->mask();
  std::vector<size_t> posIndeces;

  int dimSrcVoc = options_->get<std::vector<int>>("dim-vocabs")[index];

  for (size_t iPos = 0; iPos < batch->at(index)->batchSize(); ++iPos) {
    posIndeces.push_back(dimSrcVoc + iPos);
  }

  int dimBatch = batch->size();
  int dimEmb = emb->shape()[1];
  int dimWords = batch->at(index)->batchSize();

  auto graph = emb->graph();

  auto xWord = reshape(rows(emb, wordIndeces), {dimBatch, dimEmb, dimWords});
  auto xPos = reshape(rows(emb, posIndeces), {dimBatch, dimEmb, dimWords});
  auto x = xWord + xPos;
  auto xMask = graph->constant(shape={dimBatch, 1, dimWords},
                                init=inits::from_vector(mask));
  return std::make_tuple(x, xMask);
}

Ptr<DecoderState> ConvolutionalDecoder::startState(Ptr<EncoderState> encState) {
  using namespace keywords;

  auto meanContext =
    weighted_average(std::static_pointer_cast<ConvEncoderState>(encState)->getConvContext(),
                     encState->getMask(),
                     axis=2);

  bool layerNorm = options_->get<bool>("normalize");
  auto start = Dense("ff_state",
                     options_->get<int>("dim-rnn"),
                     activation=act::tanh,
                     normalize=layerNorm)(meanContext);
  std::vector<Expr> startStates(options_->get<size_t>("layers-dec"), start);
  return New<DecoderStateS2S>(startStates, nullptr, encState);
}

Ptr<DecoderState>
ConvolutionalDecoder::step(Expr embeddings, Ptr<DecoderState> state, bool single) {
  using namespace keywords;

  int dimTrgVoc = options_->get<std::vector<int>>("dim-vocabs").back();
  int dimTrgEmb = options_->get<int>("dim-emb");
  int dimDecState = options_->get<int>("dim-rnn");
  bool layerNorm = options_->get<bool>("normalize");
  bool skipDepth = options_->get<bool>("skip");
  size_t decoderLayers = options_->get<size_t>("layers-dec");

  float dropoutRnn = inference_ ? 0 : options_->get<float>("dropout-rnn");
  float dropoutTrg = inference_ ? 0 : options_->get<float>("dropout-trg");

  auto graph = embeddings->graph();

  if(dropoutTrg) {
    int trgWords = embeddings->shape()[2];
    auto trgWordDrop = graph->dropout(dropoutTrg, {1, 1, trgWords});
    embeddings = dropout(embeddings, mask=trgWordDrop);
  }

  if (!attention_)
    attention_ = New<GlobalAttention>("decoder",
                                      state->getEncoderState(),
                                      std::static_pointer_cast<ConvEncoderState>(state->getEncoderState())->getConvContext(),
                                      dimDecState,
                                      dropout_prob=dropoutRnn,
                                      normalize=layerNorm);
  RNN<CGRU> rnnL1(graph, "decoder",
                  dimTrgEmb, dimDecState,
                  attention_,
                  dropout_prob=dropoutRnn,
                  normalize=layerNorm);

  auto stateS2S = std::dynamic_pointer_cast<DecoderStateS2S>(state);
  auto stateL1 = rnnL1(embeddings, stateS2S->getStates()[0]);
  auto alignedContext = single ?
    rnnL1.getCell()->getLastContext() :
    rnnL1.getCell()->getContexts();

  std::vector<Expr> statesOut;
  statesOut.push_back(stateL1);

  Expr outputLn;
  if(decoderLayers > 1) {
    std::vector<Expr> statesIn;
    for(int i = 1; i < stateS2S->getStates().size(); ++i)
      statesIn.push_back(stateS2S->getStates()[i]);

    std::vector<Expr> statesLn;
    std::tie(outputLn, statesLn) = MLRNN<GRU>(graph, "decoder",
                                              decoderLayers - 1,
                                              dimDecState, dimDecState,
                                              normalize=layerNorm,
                                              dropout_prob=dropoutRnn,
                                              skip=skipDepth,
                                              skip_first=skipDepth)
                                              (stateL1, statesIn);

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
                    (embeddings, outputLn, alignedContext);

  auto logitsOut = filterInfo_ ?
    DenseWithFilter("ff_logit_l2", dimTrgVoc, filterInfo_->indeces())(logitsL1) :
    Dense("ff_logit_l2", dimTrgVoc)(logitsL1);

  return New<DecoderStateS2S>(statesOut, logitsOut,
                              state->getEncoderState());
}


}  // namespace marian
