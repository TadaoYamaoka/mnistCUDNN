#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <chrono>
#include <thread>
using namespace std;

#include "nn.h"

const int gpu_num = 1;

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

int main()
{
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

	NN nn;

	nn.load_model("../chainer/model");

	// read all test images
	float *images = new float[IMAGE_H * IMAGE_W * numberOfImages.val];
	for (int i = 0; i < numberOfImages.val; i++) {
		for (int h = 0; h < IMAGE_H; h++) {
			for (int w = 0; w < IMAGE_W; w++) {
				// pixel(unsigned byte)
				unsigned char pixel;
				ifs.read((char*)&pixel, 1);

				images[i * h * w] = float(pixel) / 255.0f;
			}
		}
	}

	thread th[gpu_num];
	int itr[gpu_num];
	chrono::system_clock::duration elapsed_time[gpu_num];

	for (int gpu = 0; gpu < gpu_num; gpu++) {
		elapsed_time[gpu] = chrono::system_clock::duration::zero();
		int& itr_th = itr[gpu];
		auto& elapsed_time_th = elapsed_time[gpu];

		th[gpu] = thread([&itr_th, &elapsed_time_th, images, gpu, &nn, numberOfImages] {
			NN::y_t y;

			checkCudaErrors(cudaSetDevice(gpu));

			const int itr_num = numberOfImages.val / batch_size / gpu_num;
			for (itr_th = 0; itr_th < itr_num; itr_th++)
			{
				float* x = images + (IMAGE_H * IMAGE_W) * itr_th;

				auto start = chrono::system_clock::now();
				nn.foward(x, (float*)y);
				elapsed_time_th += chrono::system_clock::now() - start;
			}
		});
	}

	for (int gpu = 0; gpu < gpu_num; gpu++) {
		th[gpu].join();

		cout << "gpu:" << gpu << endl;
		cout << itr[gpu] << " iterations" << endl;
		auto msec = chrono::duration_cast<std::chrono::milliseconds>(elapsed_time[gpu]).count();
		cout << msec << " [ms]" << endl;
	}

	return 0;
}