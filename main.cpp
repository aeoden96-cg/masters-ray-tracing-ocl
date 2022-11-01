// OpenCL based simple sphere path tracer by Sam Lapere, 2016
// based on smallpt by Kevin Beason 
// http://raytracey.blogspot.com 





#include <chrono>
#include "OCL.h"





Settings settings = { 720, 480, 5, 100 };


int main(int argc, char** argv){
	bool info = false;

    OCL ocl(settings);
	// console arguments
	std::vector<std::string> args;
	std::copy(argv + 1, argv + argc, std::back_inserter(args));

	if (args.size() > 0){
		if (args[0] == "info") info = true;
		else{
			cout << "Invalid argument! Valid arguments are: info" << endl;
			return 0;
		}
	}

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	// initialise OpenCL
	ocl.initOpenCL(info);
    ocl.initSceneSpheres();
	ocl.initScenePlanes();
	ocl.load2Gpu();
	ocl.render();

	// save image
    ocl.saveImage();
	cout << "Saved image to 'opencl_raytracer.ppm'" << endl;

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
}

