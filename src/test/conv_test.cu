#include <iostream>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda.h>
#include <cudnn.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      return EXIT_FAILURE;}} while(0)

#define CUDNN_CALL(x) do { if((x) != CUDNN_STATUS_SUCCESS) { \
      printf("Error (%s) at %s:%d\n",cudnnGetErrorString(x),__FILE__,__LINE__);     \
      return EXIT_FAILURE;}} while(0)

int main() {
  std::cerr << "Creating CUDNN handle...";
  cudnnHandle_t cudnnHandle;
  cudnnCreate(&cudnnHandle);
  std::cerr << " DONE\n";

  int numWords = 20;
  int dimWord = 100;
  int numBatches = 1;
  int numLayers = 1;

  std::cerr << "Allocating sample data (embeddings)...";
  std::vector<float> embeddings(numWords * dimWord);
  for(int i = 0; i < numWords * dimWord; ++i) {
    embeddings[i] = i;
  }

  float* h_emb;
  CUDA_CALL( cudaMalloc(&h_emb, sizeof(float) * numWords * dimWord) );
  CUDA_CALL( cudaMemcpy(h_emb, embeddings.data(), sizeof(float) * numWords * dimWord, cudaMemcpyHostToDevice) );

  cudnnTensorDescriptor_t embTensor;
  CUDNN_CALL( cudnnCreateTensorDescriptor(&embTensor) );
  CUDNN_CALL( cudnnSetTensor4dDescriptor(embTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             numBatches, numLayers, dimWord, numWords) );
  std::cerr << " DONE\n";

  std::cerr << "Creating and setting Conv Descriptor...";
  cudnnConvolutionDescriptor_t convDesc;
  CUDNN_CALL( cudnnCreateConvolutionDescriptor(&convDesc) );
  CUDNN_CALL( cudnnSetConvolution2dDescriptor(convDesc,
      0, 1,  // padding
      1, 1,  // strides
      1, 1,  // upscales
      CUDNN_CONVOLUTION) );

  std::cerr << " DONE\n";



  std::cerr << "Creating and setting Filters...";
  cudnnFilterDescriptor_t filterDesc;
  CUDNN_CALL( cudnnCreateFilterDescriptor(&filterDesc) );

  int numFilters = 2;
  int filterWidth = 3;

  CUDNN_CALL( cudnnSetFilter4dDescriptor(
        filterDesc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        numFilters, 1,
        dimWord, filterWidth
  ) );
  std::vector<float> filters(numFilters * dimWord * filterWidth);
  for (auto& v: filters) v = 1.0f;
  float* d_filters;
  CUDA_CALL( cudaMalloc(&d_filters, numFilters * dimWord * filterWidth * sizeof(float)) );
  CUDA_CALL( cudaMemcpy(d_filters, filters.data(), filters.size() * sizeof(float), cudaMemcpyHostToDevice ) );

  std::cerr << " DONE\n";

  std::cerr << "Allocating output memory...";

  int outputN, outputC, outputH, outputW;
  CUDNN_CALL( cudnnGetConvolution2dForwardOutputDim(convDesc, embTensor, filterDesc,
              &outputN, &outputC, &outputH, &outputW) );
  std::vector<float> output(outputN * outputC * outputH * outputW);
  float* d_output;
  CUDA_CALL( cudaMalloc(&d_output, outputN * outputC * outputH * outputW * sizeof(float)) );
  cudnnTensorDescriptor_t outputDesc;
  CUDNN_CALL( cudnnCreateTensorDescriptor(&outputDesc) );
  CUDNN_CALL( cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             outputN, outputC, outputH, outputW) );
  std::cerr << " DONE\n";
  std::cerr << "output dim: " << outputN << " " << outputC << " " << outputH << " " << outputW << std::endl;

  float alpha = 1.0f;
  float beta = 0.0f;

  std::cerr << "Start finding conv algorithm...";

  int maxNumAlgos = 3;
  cudnnConvolutionFwdAlgoPerf_t algos[maxNumAlgos];
  int numAlgos = -1;
  CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(cudnnHandle, embTensor, filterDesc, convDesc, outputDesc,
      maxNumAlgos, &numAlgos, algos) );
  std::cerr << " DONE\n";

  std::cerr << "returned algo results: " << numAlgos << std::endl;

  for (int i = 0; i < numAlgos; ++i) {
    std::cerr
      << "Algorithm: " << int(algos[i].algo) << "\t"
      << "Status: " << cudnnGetErrorString(algos[i].status) << "\t"
      << "memory: " << algos[i].memory << "\t"
      << std::endl;
  }

  cudnnConvolutionFwdAlgoPerf_t& bestAlgo = algos[0];


  float* d_workspace;
  int workspaceSize = bestAlgo.memory;
  CUDA_CALL( cudaMalloc(&d_workspace, workspaceSize) );

  CUDA_CALL( cudaSetDevice(0) );

  std::cerr << "Start forward...";

  CUDNN_CALL( cudnnConvolutionForward(
                                cudnnHandle,
                                &alpha,
                                embTensor,
                                h_emb,
                                filterDesc,
                                d_filters,
                                convDesc,
                                bestAlgo.algo,
                                d_workspace,
                                workspaceSize,
                                &beta,
                                outputDesc,
                                d_output
  ) );

  std::cerr << " DONE\n";

  std::cerr << "Printing results: " << std::endl;

  CUDA_CALL( cudaMemcpy(output.data(), d_output, sizeof(float) * outputN * outputC * outputH * outputW,
        cudaMemcpyDeviceToHost) );

  for (size_t i = 0; i < numFilters; ++i) {
    for (size_t j = 0; j < outputH * outputW; ++j) {
      std::cerr << output[i * outputH * outputW + j] << " ";
    }
    std::cerr << std::endl;
  }

  return 0;
}
