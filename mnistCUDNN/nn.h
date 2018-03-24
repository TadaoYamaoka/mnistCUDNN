#pragma once

#include "layers.h"

const int IMAGE_H = 28;
const int IMAGE_W = 28;

const int batch_size = 100;


class NN {
public:
	typedef float x_t[batch_size][1][IMAGE_H][IMAGE_W];
	typedef float y_t[batch_size][10];

	NN();
	~NN();

	void load_model(const char* filename);

	void foward(x_t x, y_t y);

private:
	static CudnnHandle cudnnHandle;
	static CublasHandle cublasHandle;
	static const int k = 16;
	static const int fcl = 256;

	ConvLayer<k, 1, 3, 1> conv1;
	Bias<k, 1, 1> bias1;
	ConvLayer<k, k, 3, 1> conv2;
	Bias<k, 1, 1> bias2;
	Linear<7 * 7 * k, fcl> l3;
	Bias<fcl, 1, 1> bias3;
	Linear<fcl, 10> l4;
	Bias<10, 1, 1> bias4;

	ReLU relu;
	MaxPooling2D<2> max_pooling_2d;

	CudnnTensorDescriptor xDesc;
	CudnnTensorDescriptor h1Desc;
	CudnnTensorDescriptor h2Desc;
	CudnnTensorDescriptor h3Desc;
	CudnnTensorDescriptor h4Desc;
	CudnnTensorDescriptor h5Desc;
	CudnnTensorDescriptor yDesc;

	float* x_dev;
	float* h1_dev;
	float* h2_dev;
	float* h3_dev;
	float* h4_dev;
	float* h5_dev;
	float* y_dev;
};