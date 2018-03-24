#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <algorithm>
using namespace std;

#include "nn.h"

int gpu_num = 1;

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
	if (argc > 1) {
		gpu_num = atoi(argv[1]);
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

	vector<thread> th(gpu_num);
	vector<int> itr(gpu_num);
	vector<chrono::system_clock::duration> elapsed_time(gpu_num);
	vector<chrono::system_clock::time_point> start_time(gpu_num);
	vector<chrono::system_clock::time_point> end_time(gpu_num);
	condition_variable cond;
	mutex mtx;
	bool ready = false;

	for (int gpu = 0; gpu < gpu_num; gpu++) {
		elapsed_time[gpu] = chrono::system_clock::duration::zero();
		int& itr_th = itr[gpu];
		auto& elapsed_time_th = elapsed_time[gpu];
		auto& start_time_th = start_time[gpu];
		auto& end_time_th = end_time[gpu];

		th[gpu] = thread([&itr_th, &elapsed_time_th, &start_time_th, &end_time_th, images, gpu, numberOfImages, &cond, &mtx, &ready] {
			checkCudaErrors(cudaSetDevice(gpu));

			NN nn;
			nn.load_model("../chainer/model");

			NN::y_t y;

			unique_lock<mutex> lk(mtx);
			cond.wait(lk, [&ready] { return ready; });
			start_time_th = chrono::system_clock::now();

			const int itr_num = numberOfImages.val / batch_size / gpu_num;
			for (int epoch = 0; epoch < 100; epoch++) {
				for (itr_th = 0; itr_th < itr_num; itr_th++)
				{
					float* x = images + (IMAGE_H * IMAGE_W) * (itr_th + itr_num * gpu);

					auto start = chrono::system_clock::now();
					nn.foward(x, (float*)y);
					elapsed_time_th += chrono::system_clock::now() - start;
				}
			}

			end_time_th = chrono::system_clock::now();
		});
	}

	// start measurement
	{
		lock_guard<mutex> lk(mtx);
		ready = true;
	}
	cond.notify_all();

	for (int gpu = 0; gpu < gpu_num; gpu++) {
		th[gpu].join();
	}

	for (int gpu = 0; gpu < gpu_num; gpu++) {
		cout << "gpu:" << gpu << endl;
		cout << itr[gpu] << " iterations" << endl;
		auto msec = chrono::duration_cast<std::chrono::milliseconds>(elapsed_time[gpu]).count();
		cout << msec << " [ms]" << endl;
	}
	auto start_total = *min_element(start_time.begin(), start_time.end());
	auto end_total = *max_element(end_time.begin(), end_time.end());
	auto elapsed_time_total = end_total - start_total;
	cout << "total time = " << chrono::duration_cast<std::chrono::milliseconds>(elapsed_time_total).count() << " [ms]" << endl;

	return 0;
}