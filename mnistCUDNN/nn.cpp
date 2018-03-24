#include "nn.h"
#include "npz.h"

CudnnHandle NN::cudnnHandle;
CublasHandle NN::cublasHandle;

NN::NN()
{
	conv1.get_xdesc(xDesc, batch_size, IMAGE_H, IMAGE_W);

	const int h1_h = conv1.get_yh(IMAGE_H);
	const int h1_w = conv1.get_yw(IMAGE_W);
	conv1.get_ydesc(h1Desc, batch_size, h1_h, h1_w);

	const int h3_h = max_pooling_2d.get_yh(h1_h);
	const int h3_w = max_pooling_2d.get_yw(h1_w);
	conv3.get_xdesc(h3Desc, batch_size, h3_h, h3_w);

	const int h4_h = conv3.get_yh(h3_h);
	const int h4_w = conv3.get_yw(h3_w);
	conv3.get_ydesc(h4Desc, batch_size, h4_h, h4_w);

	const int h5_h = max_pooling_2d.get_yh(h4_h);
	const int h5_w = max_pooling_2d.get_yw(h4_w);
	max_pooling_2d.get_desc(h5Desc, batch_size, k, h5_h, h5_w);

	l4.get_ydesc(h6Desc, batch_size);
	l5.get_ydesc(yDesc, batch_size);

	// init conv layers
	conv1.init(cudnnHandle, xDesc, h1Desc);
	conv2.init(cudnnHandle, h1Desc, h1Desc);
	conv3.init(cudnnHandle, h3Desc, h4Desc);

	// malloc
	checkCudaErrors(cudaMalloc((void**)&x_dev, conv1.get_xsize(batch_size, IMAGE_H, IMAGE_W)));
	checkCudaErrors(cudaMalloc((void**)&h1_dev, conv1.get_ysize(batch_size, h1_h, h1_w)));
	checkCudaErrors(cudaMalloc((void**)&h1_bn_dev, conv1.get_ysize(batch_size, h1_h, h1_w)));
	checkCudaErrors(cudaMalloc((void**)&h2_dev, conv2.get_ysize(batch_size, h1_h, h1_w)));
	checkCudaErrors(cudaMalloc((void**)&h2_bn_dev, conv2.get_ysize(batch_size, h1_h, h1_w)));
	checkCudaErrors(cudaMalloc((void**)&h3_dev, conv3.get_xsize(batch_size, h3_h, h3_w)));
	checkCudaErrors(cudaMalloc((void**)&h4_dev, conv3.get_ysize(batch_size, h4_h, h4_w)));
	checkCudaErrors(cudaMalloc((void**)&h5_dev, batch_size * k * h5_h * h5_w * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&h6_dev, batch_size * fcl * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&y_dev, batch_size * 10 * sizeof(float)));
}

NN::~NN() {
	checkCudaErrors(cudaFree(x_dev));
	checkCudaErrors(cudaFree(h1_dev));
	checkCudaErrors(cudaFree(h1_bn_dev));
	checkCudaErrors(cudaFree(h2_dev));
	checkCudaErrors(cudaFree(h2_bn_dev));
	checkCudaErrors(cudaFree(h3_dev));
	checkCudaErrors(cudaFree(h4_dev));
	checkCudaErrors(cudaFree(h5_dev));
	checkCudaErrors(cudaFree(h6_dev));
	checkCudaErrors(cudaFree(y_dev));
}

void NN::load_model(const char* filepath)
{
	// load nn params
	ParamMap params;
	load_npz(filepath, params);

	conv1.set_param(params["conv1/W.npy"].data);
	bias1.set_bias(params["conv1/b.npy"].data);
	conv2.set_param(params["conv2/W.npy"].data);
	bias2.set_bias(params["conv2/b.npy"].data);
	conv3.set_param(params["conv3/W.npy"].data);
	bias3.set_bias(params["conv3/b.npy"].data);
	l4.set_param(params["l4/W.npy"].data);
	bias4.set_bias(params["l4/b.npy"].data);
	l5.set_param(params["l5/W.npy"].data);
	bias5.set_bias(params["l5/b.npy"].data);
	bn1.set_param(params["bn1/gamma.npy"].data, params["bn1/beta.npy"].data, params["bn1/avg_mean.npy"].data, params["bn1/avg_var.npy"].data);
	bn2.set_param(params["bn2/gamma.npy"].data, params["bn2/beta.npy"].data, params["bn2/avg_mean.npy"].data, params["bn2/avg_var.npy"].data);
}

void NN::foward(float* x, float* y)
{
	// input
	checkCudaErrors(cudaMemcpy(x_dev, x, sizeof(x_t), cudaMemcpyHostToDevice));

	// conv1
	conv1(cudnnHandle, xDesc, x_dev, h1Desc, h1_dev);
	bias1(cudnnHandle, h1Desc, h1_dev);
	bn1(cudnnHandle, h1Desc, h1_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);

	// resnet block
	conv2(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bias2(cudnnHandle, h1Desc, h2_dev);
	bn2(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);

	add(cudnnHandle, h1Desc, h1_bn_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	max_pooling_2d(cudnnHandle, h1Desc, h2_bn_dev, h3Desc, h3_dev);

	// conv3
	conv3(cudnnHandle, h3Desc, h3_dev, h4Desc, h4_dev);
	bias3(cudnnHandle, h4Desc, h4_dev);
	relu(cudnnHandle, h4Desc, h4_dev);
	max_pooling_2d(cudnnHandle, h4Desc, h4_dev, h5Desc, h5_dev);

	// fcl
	l4(cublasHandle, batch_size, h5_dev, h6_dev);
	bias4(cudnnHandle, h6Desc, h6_dev);
	relu(cudnnHandle, h6Desc, h6_dev);
	l5(cublasHandle, batch_size, h6_dev, y_dev);
	bias5(cudnnHandle, yDesc, y_dev);

	// output
	checkCudaErrors(cudaMemcpy(y, y_dev, sizeof(y_t), cudaMemcpyDeviceToHost));
}