#pragma once

#include "cudnn_wrapper.h"

extern const __half _half_zero;
extern const __half _half_one;

template<const int k, const int c, const int fsize, const int pad, const int stride = 1>
class ConvLayer {
public:
	ConvLayer() : W(nullptr), workSpace(nullptr) {
		const size_t size = c * k * fsize * fsize;
		checkCudaErrors(cudaMalloc((void**)&W, size * sizeof(__half)));
	}
	~ConvLayer() {
		checkCudaErrors(cudaFree(W));
		checkCudaErrors(cudaFree(workSpace));
	}

	void init(cudnnHandle_t handle, cudnnTensorDescriptor_t xDesc, cudnnTensorDescriptor_t yDesc) {
		checkCUDNN(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, k, c, fsize, fsize));
		checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, pad, pad, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF));
		checkCUDNN(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
		int returnedAlgoCount;
		checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(handle, xDesc, wDesc, convDesc, yDesc, 1, &returnedAlgoCount, &algo_perf));
		checkCudaErrors(cudaMalloc(&workSpace, algo_perf.memory));
	}

	int get_yh(const int h) {
		return (h + 2 * pad - fsize) / stride + 1;
	}

	int get_yw(const int w) {
		return (w + 2 * pad - fsize) / stride + 1;
	}

	void get_xdesc(cudnnTensorDescriptor_t xDesc, const int n, const int h, const int w) {
		checkCUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n, c, h, w));
	}

	void get_ydesc(cudnnTensorDescriptor_t yDesc, const int n, const int h, const int w) {
		checkCUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n, k, h, w));
	}

	int get_xsize(const int n, const int h, const int w) {
		return n * c * h * w * sizeof(__half);
	}

	int get_ysize(const int n, const int h, const int w) {
		return n * k * h * w * sizeof(__half);
	}

	void set_param(float* data) {
		const size_t size = c * k * fsize * fsize;
		__half* tmp = new __half[size];
		for (size_t i = 0; i < size; i++)
			tmp[i] = __float2half(data[i]);
		checkCudaErrors(cudaMemcpy(W, tmp, size * sizeof(__half), cudaMemcpyHostToDevice));
		delete[] tmp;
	}

	void operator() (cudnnHandle_t handle, cudnnTensorDescriptor_t xDesc, __half* x, cudnnTensorDescriptor_t yDesc, __half* y) {
		const __half alpha = _half_one;
		const __half beta = _half_zero;
		checkCUDNN(cudnnConvolutionForward(handle, &alpha, xDesc, x, wDesc, W, convDesc, algo_perf.algo, workSpace, algo_perf.memory, &beta, yDesc, y));
	}

private:
	CudnnFilterDescriptor wDesc;
	CudnnConvolutionDescriptor convDesc;
	cudnnConvolutionFwdAlgoPerf_t algo_perf;
	__half* W;
	void* workSpace;
};

template<const int c, const int h, const int w>
class Bias {
public:
	Bias() : b(nullptr) {
		checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, c, h, w));
		const size_t size = c * h * w;
		checkCudaErrors(cudaMalloc((void**)&b, size * sizeof(__half)));
	}
	~Bias() {
		checkCudaErrors(cudaFree(b));
	}

	void set_bias(float* data) {
		const size_t size = c * h * w;
		__half* tmp = new __half[size];
		for (size_t i = 0; i < size; i++)
			tmp[i] = __float2half(data[i]);
		checkCudaErrors(cudaMemcpy(b, tmp, size * sizeof(__half), cudaMemcpyHostToDevice));
		delete[] tmp;
	}

	void operator() (cudnnHandle_t handle, cudnnTensorDescriptor_t xDesc, __half* x) {
		const __half alpha = _half_one;
		const __half beta = _half_one;
		checkCUDNN(cudnnAddTensor(handle, &alpha, biasTensorDesc, b, &beta, xDesc, x));
	}

private:
	CudnnTensorDescriptor biasTensorDesc;
	__half *b;
};

class ReLU {
public:
	ReLU() {
		checkCUDNN(cudnnSetActivationDescriptor(activDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0/*reluCeiling*/));
	}

	void operator() (cudnnHandle_t handle, cudnnTensorDescriptor_t xDesc, __half* x) {
		const __half alpha = _half_one;
		const __half beta = _half_zero;
		checkCUDNN(cudnnActivationForward(handle, activDesc, &alpha, xDesc, x, &beta, xDesc, x));
	}

private:
	CudnnActivationDescriptor activDesc;
};

template<const int k, const int n>
class Linear {
public:
	Linear() : W(nullptr) {
		const size_t size = k * n;
		checkCudaErrors(cudaMalloc((void**)&W, size * sizeof(__half)));
	}
	~Linear() {
		checkCudaErrors(cudaFree(W));
	}

	void get_xdesc(cudnnTensorDescriptor_t xDesc, const int m) {
		checkCUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, m, k, 1, 1));
	}

	void get_ydesc(cudnnTensorDescriptor_t yDesc, const int m) {
		checkCUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, m, n, 1, 1));
	}

	void set_param(float* data) {
		const size_t size = k * n;
		__half* tmp = new __half[size];
		for (size_t i = 0; i < size; i++)
			tmp[i] = __float2half(data[i]);
		checkCudaErrors(cudaMemcpy(W, tmp, size * sizeof(__half), cudaMemcpyHostToDevice));
		delete[] tmp;
	}

	void operator() (cublasHandle_t handle, const int m, __half* x, __half* y) {
		const __half alpha = _half_one;
		const __half beta = _half_zero;
		// C = ƒ¿ op ( A ) op ( B ) + ƒÀ C
		// op ( A ) m ~ k , op ( B ) k ~ n and C m ~ n
		checkCublasErrors(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, W, CUDA_R_16F, k, x, CUDA_R_16F, k, &beta, y, CUDA_R_16F, n, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	}

private:
	__half* W;
};

template<const int window, const int stride = window, const int pad = 0>
class MaxPooling2D {
public:
	MaxPooling2D() {
		checkCUDNN(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, window, window, pad, pad, stride, stride));
	}

	int get_yh(const int h) {
		return (h + 2 * pad - window) / stride + 1;
	}

	int get_yw(const int w) {
		return (w + 2 * pad - window) / stride + 1;
	}

	void get_desc(cudnnTensorDescriptor_t desc, const int n, const int c, const int h, const int w) {
		checkCUDNN(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n, c, h, w));
	}

	void operator() (cudnnHandle_t handle, cudnnTensorDescriptor_t xDesc, __half* x, cudnnTensorDescriptor_t yDesc, __half* y) {
		const __half alpha = _half_one;
		const __half beta = _half_zero;
		checkCUDNN(cudnnPoolingForward(handle, poolingDesc, &alpha, xDesc, x, &beta, yDesc, y));
	}

private:
	CudnnPoolingDescriptor poolingDesc;
};

class Softmax {
public:
	void get_desc(cudnnTensorDescriptor_t desc, const int n, const int c) {
		checkCUDNN(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n, c, 1, 1));
	}

	void operator() (cudnnHandle_t handle, cudnnTensorDescriptor_t xDesc, __half* x) {
		const __half alpha = _half_one;
		const __half beta = _half_zero;
		checkCUDNN(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, xDesc, x, &beta, xDesc, x));
	}
};

template<const int k>
class BatchNormalization {
public:
	BatchNormalization() : bnScale(nullptr), bnBias(nullptr), estimatedMean(nullptr), estimatedVariance(nullptr) {
		const size_t size = k;
		checkCudaErrors(cudaMalloc((void**)&bnScale, size * sizeof(__half)));
		checkCudaErrors(cudaMalloc((void**)&bnBias, size * sizeof(__half)));
		checkCudaErrors(cudaMalloc((void**)&estimatedMean, size * sizeof(__half)));
		checkCudaErrors(cudaMalloc((void**)&estimatedVariance, size * sizeof(__half)));
	}
	~BatchNormalization() {
		checkCudaErrors(cudaFree(bnScale));
		checkCudaErrors(cudaFree(bnBias));
		checkCudaErrors(cudaFree(estimatedMean));
		checkCudaErrors(cudaFree(estimatedVariance));
	}

	void operator() (cudnnHandle_t handle, cudnnTensorDescriptor_t xDesc, __half* x, __half* y) {
		const __half alpha = _half_one;
		const __half beta = _half_zero;
		const double eps = 2e-5;
		checkCUDNN(cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVarDesc, xDesc, CUDNN_BATCHNORM_SPATIAL));
		checkCUDNN(cudnnBatchNormalizationForwardInference(handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, xDesc, x, xDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, eps));
	}

	void set_param(float* bnScale, float *bnBias, float *estimatedMean, float *estimatedVariance) {
		const size_t size = k;
		__half* tmp_bnScale = new __half[size];
		__half* tmp_bnBias = new __half[size];
		__half* tmp_estimatedMean = new __half[size];
		__half* tmp_estimatedVariance = new __half[size];
		for (size_t i = 0; i < size; i++) {
			tmp_bnScale[i] = __float2half(bnScale[i]);
			tmp_bnBias[i] = __float2half(bnBias[i]);
			tmp_estimatedMean[i] = __float2half(estimatedMean[i]);
			tmp_estimatedVariance[i] = __float2half(estimatedVariance[i]);
		}
		checkCudaErrors(cudaMemcpy(this->bnScale, tmp_bnScale, size * sizeof(__half), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(this->bnBias, tmp_bnBias, size * sizeof(__half), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(this->estimatedMean, tmp_estimatedMean, size * sizeof(__half), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(this->estimatedVariance, tmp_estimatedVariance, size * sizeof(__half), cudaMemcpyHostToDevice));
	}

private:
	CudnnTensorDescriptor bnScaleBiasMeanVarDesc;
	__half *bnScale;
	__half *bnBias;
	__half *estimatedMean;
	__half *estimatedVariance;
};

class Add {
public:
	void operator() (cudnnHandle_t handle, cudnnTensorDescriptor_t xDesc, __half* x, __half* y) {
		const __half alpha = _half_one;
		const __half beta = _half_one;
		checkCUDNN(cudnnAddTensor(handle, &alpha, xDesc, x, &beta, xDesc, y));
	}
};