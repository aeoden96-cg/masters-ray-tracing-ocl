// OpenCL based simple sphere path tracer by Sam Lapere, 2016
// based on smallpt by Kevin Beason 
// http://raytracey.blogspot.com 

#include <iostream>
#include <vector>

#include "OCL.h"
#include <chrono>

using namespace std;
using namespace cl;

const int image_width = 1024;
const int image_height = 720;

const int sphere_count = 9;
const int plane_count = 1;

int main(int argc, char** argv){
	bool info = false;

	// console arguments
	std::vector<std::string> args;
	std::copy(argv + 1, argv + argc, std::back_inserter(args));

	if (!args.empty()){
		if (args[0] == "info") info = true;
		else{
			cout << "Invalid argument! Valid arguments are: info" << endl;
			return 0;
		}
	}

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    OCL ocl(info, image_width,image_height);

	ocl.initScene(sphere_count);
	ocl.initScenePlanes(plane_count);

    ocl.setGPUArgs();
    ocl.render();

	// save image
	ocl.saveImage();
	cout << "Saved image to 'opencl_raytracer.ppm'" << endl;

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
}
