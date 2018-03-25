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

	void foward(float* x, float* y);

private:
	CudnnHandle cudnnHandle;
	CublasHandle cublasHandle;
	static const int k = 16;
	static const int fcl = 256;

	ConvLayer<k, 1, 3, 1> conv1;
	Bias<k, 1, 1> bias1;
	ConvLayer<k, k, 3, 1> conv2;
	Bias<k, 1, 1> bias2;
	ConvLayer<k, k, 3, 1> conv3;
	Bias<k, 1, 1> bias3;
	Linear<7 * 7 * k, fcl> l4;
	Bias<fcl, 1, 1> bias4;
	Linear<fcl, 10> l5;
	Bias<10, 1, 1> bias5;
	BatchNormalization<k> bn1;
	BatchNormalization<k> bn2;

	ReLU relu;
	MaxPooling2D<2> max_pooling_2d;
	Add add;

	CudnnTensorDescriptor xDesc;
	CudnnTensorDescriptor h1Desc;
	CudnnTensorDescriptor h2Desc;
	CudnnTensorDescriptor h3Desc;
	CudnnTensorDescriptor h4Desc;
	CudnnTensorDescriptor h5Desc;
	CudnnTensorDescriptor h6Desc;
	CudnnTensorDescriptor yDesc;

	float* x_dev;
	float* h1_dev;
	float* h1_bn_dev;
	float* h2_dev;
	float* h2_bn_dev;
	float* h3_dev;
	float* h4_dev;
	float* h5_dev;
	float* h6_dev;
	float* y_dev;
};