﻿#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <chrono>
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

int main()
{
	//showDevices();

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

	NN::y_t y;

	auto start0 = chrono::system_clock::now();
	auto elapsed_time = start0 - start0;
	int itr;
	for (itr = 0; itr < numberOfImages.val / batch_size; itr++)
	{
		// make minibatch
		NN::x_t x;
		for (int i = 0; i < batch_size; i++) {
			for (int h = 0; h < IMAGE_H; h++) {
				for (int w = 0; w < IMAGE_W; w++) {
					// pixel(unsigned byte)
					unsigned char pixel;
					ifs.read((char*)&pixel, 1);

					x[i][0][h][w] = float(pixel) / 255.0f;
				}
			}
		}

		auto start = chrono::system_clock::now();
		nn.foward(x, y);
		elapsed_time += chrono::system_clock::now() - start;

		/*for (int i = 0; i < batch_size; i++) {
			for (int c = 0; c < 10; c++) {
				if (c > 0)
					cout << "\t";
				cout << y[i][c];
			}
			cout << endl;
		}*/
	}
	cout << itr << " iterations" << endl;

	auto msec = chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count();
	cout << msec << " [ms]" << endl;

	return 0;
}