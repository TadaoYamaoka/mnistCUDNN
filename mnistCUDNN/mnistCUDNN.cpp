#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <algorithm>
using namespace std;

#include "nn.h"

static void  showDevices(void)
{
	int totalDevices;
	checkCudaErrors(cudaGetDeviceCount(&totalDevices));
	cout << "There are " << totalDevices << " CUDA capable devices on your machine :\n";
	for (int i = 0; i < totalDevices; i++) {
		struct cudaDeviceProp prop;
		checkCudaErrors(cudaGetDeviceProperties(&prop, i));
		cout << "device " << i
			<< " sms " << prop.multiProcessorCount
			<< " Capabilities " << prop.major << "." << prop.minor
			<< ", SmClock " << (float)prop.clockRate*1e-3 << " Mhz"
			<< ", MemSize (Mb) " << (int)(prop.totalGlobalMem / (1024 * 1024))
			<< ", MemClock " << (float)prop.memoryClockRate*1e-3 << " Mhz"
			<< ", Ecc=" << prop.ECCEnabled
			<< ", boardGroupID=" << prop.multiGpuBoardGroupID << endl;
	}
}

// All the integers in the files are stored in the MSB first (high endian) format used by most non-Intel processors.
struct msb_unsigned_int_t {
	union {
		unsigned char byte[4];
		unsigned int val;
	};
};
ifstream& operator >> (ifstream& is, msb_unsigned_int_t& d) {
	is.read((char*)&d.byte[3], 1);
	is.read((char*)&d.byte[2], 1);
	is.read((char*)&d.byte[1], 1);
	is.read((char*)&d.byte[0], 1);
	return is;
}

int main(int argc, char *argv[])
{
	int gpu_id = 0;

	if (argc > 1) {
		gpu_id = atoi(argv[1]);
	}

	showDevices();

	// read mnist data
	ifstream ifs("data/t10k-images.idx3-ubyte", ios::in | ios::binary);

	// magic number(32 bit integer)
	msb_unsigned_int_t magic_number;
	ifs >> magic_number;
	if (magic_number.val != 2051) {
		cerr << "illegal magic number" << endl;
		return 1;
	}
	// number of images(32 bit integer)
	msb_unsigned_int_t numberOfImages;
	ifs >> numberOfImages;
	// number of rows(32 bit integer)
	msb_unsigned_int_t rows;
	ifs >> rows;
	// number of columns(32 bit integer)
	msb_unsigned_int_t columns;
	ifs >> columns;

	// read all test images
	__half *images = new __half[IMAGE_H * IMAGE_W * numberOfImages.val];
	for (int i = 0; i < numberOfImages.val; i++) {
		for (int h = 0; h < IMAGE_H; h++) {
			for (int w = 0; w < IMAGE_W; w++) {
				// pixel(unsigned byte)
				unsigned char pixel;
				ifs.read((char*)&pixel, 1);

				images[i * h * w] = __float2half(pixel / 255.0f);
			}
		}
	}

	checkCudaErrors(cudaSetDevice(gpu_id));

	NN nn;
	nn.load_model("../chainer/model");

	NN::y_t y;

	long long elapsed = 0;

	const int itr_num = numberOfImages.val / batch_size;
	const int epoch_num = 20;
	auto start = chrono::system_clock::now();
	for (int epoch = 0; epoch < epoch_num; epoch++) {
		for (int itr = 0; itr < itr_num; itr++)
		{
			__half* x = images + IMAGE_H * IMAGE_W * itr * batch_size;

			nn.foward(x, (__half*)y);
		}
	}
	auto end = chrono::system_clock::now();
	elapsed += chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

	cout << "itr_num = " << itr_num << endl;
	cout << "epoch_num = " << epoch_num << endl;
	cout << "elapsed = " << elapsed << " ns" << endl;

	return 0;
}