// OpenCL based simple sphere path tracer by Sam Lapere, 2016
// based on smallpt by Kevin Beason 
// http://raytracey.blogspot.com 

# define CL_HPP_TARGET_OPENCL_VERSION 300

#include <iostream>
#include <fstream>
#include <vector>
#include <CL/opencl.hpp>
#include "Sphere.hpp"
#include <chrono>

using namespace std;
using namespace cl;




// dummy variables are required for memory alignment
// float3 is considered as float4 by OpenCL


class OCL {
public:
    OCL() {
        cpu_planes.resize(numPlanes);
        cpu_spheres.resize(numSpheres);

    }
    std::vector<Plane> cpu_planes;
    std::vector<Sphere> cpu_spheres;
    std::vector<cl_float4> cpu_output;
    CommandQueue queue;
    Device device;
    Kernel kernel;
    Context context;
    Program program;
    Buffer cl_output;
    Buffer cl_spheres;
    Buffer cl_planes;

    const int image_width = 720;
    const int image_height = 480;

    std::size_t global_work_size;
    std::size_t local_work_size;

    int samples = 100;	// number of samples per pixel
    int bounces = 8;	// number of bounces per ray

    int numSpheres = 9;
    int numPlanes = 1;

    void load2Gpu() {
// Create buffers on the OpenCL device for the image and the scene
        cl_output = Buffer(context, CL_MEM_WRITE_ONLY, image_width * image_height * sizeof(cl_float3));
        cl_spheres = Buffer(context, CL_MEM_READ_ONLY, numSpheres * sizeof(Sphere));
        queue.enqueueWriteBuffer(cl_spheres, CL_TRUE, 0, numSpheres  * sizeof(Sphere), cpu_spheres.data());
        cl_planes = Buffer(context, CL_MEM_READ_ONLY, numPlanes * sizeof(Plane));
        queue.enqueueWriteBuffer(cl_planes, CL_TRUE, 0, numPlanes * sizeof(Plane), cpu_planes.data());


        // specify OpenCL kernel arguments
        kernel.setArg(0, cl_spheres);
        kernel.setArg(1, cl_planes);
        kernel.setArg(2, image_width);
        kernel.setArg(3, image_height);
        kernel.setArg(4, numSpheres);
        kernel.setArg(5, numPlanes);
        kernel.setArg(6, samples);
        kernel.setArg(7, bounces);
        kernel.setArg(8, cl_output);
    }


    void pickPlatform(Platform& platform, const cl::vector<Platform>& platforms){

        if (platforms.size() == 1) platform = platforms[0];
        else{
            int input = 0;
            cout << "\nChoose an OpenCL platform: ";
            cin >> input;

            // handle incorrect user input
            while (input < 1 || input > platforms.size()){
                cin.clear(); //clear errors/bad flags on cin
                cin.ignore(cin.rdbuf()->in_avail(), '\n'); // ignores exact number of chars in cin buffer
                cout << "No such option. Choose an OpenCL platform: ";
                cin >> input;
            }
            platform = platforms[input - 1];
        }
    }

    void pickDevice(const cl::vector<Device>& devices){

        if (devices.size() == 1) device = devices[0];
        else{
            int input = 0;
            cout << "\nChoose an OpenCL device: ";
            cin >> input;

            // handle incorrect user input
            while (input < 1 || input > devices.size()){
                cin.clear(); //clear errors/bad flags on cin
                cin.ignore(cin.rdbuf()->in_avail(), '\n'); // ignores exact number of chars in cin buffer
                cout << "No such option. Choose an OpenCL device: ";
                cin >> input;
            }
            device = devices[input - 1];
        }
    }

    void printErrorLog(){

        // Get the error log and print to console
        string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        cerr << "Build log:" << std::endl << buildlog << std::endl;

        // Print the error log to a file
        FILE *log = fopen("errorlog.txt", "w");
        fprintf(log, "%s\n", buildlog.c_str());
        cout << "Error log saved in 'errorlog.txt'" << endl;
        exit(1);
    }

    void initOpenCL(bool info)
    {
        // Get all available OpenCL platforms (e.g. AMD OpenCL, Nvidia CUDA, Intel OpenCL)
        cl::vector<Platform> platforms;
        Platform::get(&platforms);

        if(info){
            cout << "Available OpenCL platforms : " << endl << endl;
            for (int i = 0; i < platforms.size(); i++)
                cout << "\t" << i + 1 << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << endl;
        }

        // Pick one platform
        Platform platform;
        pickPlatform(platform, platforms);

        if(info) cout << "\nUsing OpenCL platform: \t" << platform.getInfo<CL_PLATFORM_NAME>() << endl;

        // Get available OpenCL devices on platform
        cl::vector<Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        if(info){
            cout << "Available OpenCL devices on this platform: " << endl << endl;
            for (int i = 0; i < devices.size(); i++){
                cout << "\t" << i + 1 << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << endl;
                cout << "\t\tMax compute units: " << devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
                cout << "\t\tMax work group size: " << devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl << endl;
            }
        }

        // Pick one device
        pickDevice(devices);

        if(info){
            cout << "\nUsing OpenCL device: \t" << device.getInfo<CL_DEVICE_NAME>() << endl;
            cout << "\t\t\tMax compute units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
            cout << "\t\t\tMax work group size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;
        }


        // Create an OpenCL context and command queue on that device.
        context = Context(device);
        queue = CommandQueue(context, device);

        // Convert the OpenCL source code to a string
        string source;
        ifstream file("opencl_kernel.cl");
        if (!file){
            cout << "\nNo OpenCL file found!" << endl << "Exiting..." << endl;
            system("PAUSE");
            exit(1);
        }
        while (!file.eof()){
            char line[256];
            file.getline(line, 255);
            source += line;
        }

        const char* kernel_source = source.c_str();

        // Create an OpenCL program by performing runtime source compilation for the chosen device
        program = Program(context, kernel_source);
        cl_int result = program.build({ device });
        if (result) cout << "Error during compilation OpenCL code!!!\n (" << result << ")" << endl;
        if (result == CL_BUILD_PROGRAM_FAILURE) printErrorLog();

        // Create a kernel (entry point in the OpenCL source program)
        kernel = Kernel(program, "render_kernel");


        // every pixel in the image has its own thread or "work item",
        // so the total amount of work items equals the number of pixels
        global_work_size = image_width * image_height;
        local_work_size = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);


        if(info) cout << "Kernel work group size: " << local_work_size << endl;

        // Ensure the global work size is a multiple of local work size
        if (global_work_size % local_work_size != 0)
            global_work_size = (global_work_size / local_work_size + 1) * local_work_size;
    }

// convert RGB float in range [0,1] to int in range [0, 255] and perform gamma correction
    inline int toInt(float x){ return int(std::clamp(x, 0.0f, 1.0f) * 255 + .5f); }

    void saveImage(){
        // write image to PPM file, a very simple image file format
        // PPM files can be opened with IrfanView (download at www.irfanview.com) or GIMP
        FILE *f = fopen("opencl_raytracer.ppm", "w");
        fprintf(f, "P3\n%d %d\n%d\n", image_width, image_height, 255);

        // loop over all pixels, write RGB values
        for (auto& pixel : cpu_output) {
            fprintf(f, "%d %d %d ",
                    toInt(pixel.s[0]),
                    toInt(pixel.s[1]),
                    toInt(pixel.s[2]));
        }
    }

    void initScenePlanes(){
        cpu_planes[0].position = {-0.35f, -0.0f, -0.3f};
        cpu_planes[0].position2 = {0.30f, -0.0f, 0.3f};
        cpu_planes[0].color = {0.25f, 0.25f, 0.75f};
        cpu_planes[0].emission = {0, 0, 0};
        cpu_planes[0].normal = {0, 1, 0};
    }

    void initSceneSpheres(){

        // left wall
        cpu_spheres[0].radius	= 200.0f;
        cpu_spheres[0].position = {-200.6f, 0.0f, 0.0f};
        cpu_spheres[0].color    = {0.75f, 0.25f, 0.25f};
        cpu_spheres[0].emission = {0.0f, 0.0f, 0.0f};

        // right wall
        cpu_spheres[1].radius	= 200.0f;
        cpu_spheres[1].position = {200.6f, 0.0f, 0.0f};
        cpu_spheres[1].color    = {0.25f, 0.25f, 0.75f};
        cpu_spheres[1].emission = {0.0f, 0.0f, 0.0f};

        // floor
        cpu_spheres[2].radius	= 200.0f;
        cpu_spheres[2].position = {0.0f, -200.4f, 0.0f};
        cpu_spheres[2].color	= {0.9f, 0.8f, 0.7f};
        cpu_spheres[2].emission = {0.0f, 0.0f, 0.0f};

        // ceiling
        cpu_spheres[3].radius	= 200.0f;
        cpu_spheres[3].position = {0.0f, 200.4f, 0.0f};
        cpu_spheres[3].color	= {0.9f, 0.8f, 0.7f};
        cpu_spheres[3].emission = {0.0f, 0.0f, 0.0f};

        // back wall
        cpu_spheres[4].radius   = 200.0f;
        cpu_spheres[4].position = {0.0f, 0.0f, -200.4f};
        cpu_spheres[4].color    = {0.9f, 0.8f, 0.7f};
        cpu_spheres[4].emission = {0.0f, 0.0f, 0.0f};

        // front wall
        cpu_spheres[5].radius   = 200.0f;
        cpu_spheres[5].position = {0.0f, 0.0f, 202.0f};
        cpu_spheres[5].color    = {0.9f, 0.8f, 0.7f};
        cpu_spheres[5].emission = {0.0f, 0.0f, 0.0f};

        // left sphere
        cpu_spheres[6].radius   = 0.16f;
        cpu_spheres[6].position = {-0.25f, -0.24f, -0.1f};
        cpu_spheres[6].color    = {0.9f, 0.8f, 0.7f};
        cpu_spheres[6].emission = {0.0f, 0.0f, 0.0f};

        // right sphere
        cpu_spheres[7].radius   = 0.16f;
        cpu_spheres[7].position = {0.25f, -0.24f, 0.1f};
        cpu_spheres[7].color    = {0.9f, 0.8f, 0.7f};
        cpu_spheres[7].emission = {0.0f, 0.0f, 0.0f};

        // lightsource
        cpu_spheres[8].radius   = 1.0f;
        cpu_spheres[8].position = {0.0f, 1.36f, 0.0f};
        cpu_spheres[8].color    = {0.0f, 0.0f, 0.0f};
        cpu_spheres[8].emission = {9.0f, 8.0f, 6.0f};

    }



};

OCL ocl;








int main(int argc, char** argv){
	bool info = false;

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

	// allocate memory on CPU to hold the rendered image
	ocl.cpu_output.resize(ocl.image_width * ocl.image_height);

	// initialise scene


    ocl.initSceneSpheres();

	ocl.initScenePlanes();

	ocl.load2Gpu();




	cout << "Rendering started..." << endl;

	// launch the kernel
	ocl.queue.enqueueNDRangeKernel(
		ocl.kernel,
		0,
		ocl.global_work_size,
		ocl.local_work_size);

	ocl.queue.finish();

	cout << "Rendering done! \nCopying output from GPU device to host" << endl;

	// read and copy OpenCL output to CPU
	ocl.queue.enqueueReadBuffer(ocl.cl_output, CL_TRUE, 0, ocl.image_width * ocl.image_height * sizeof(cl_float3), ocl.cpu_output.data());

	// save image
    ocl.saveImage();
	cout << "Saved image to 'opencl_raytracer.ppm'" << endl;

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
}

