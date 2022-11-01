#ifndef PROJECT_OCL_H
#define PROJECT_OCL_H

# define CL_HPP_TARGET_OPENCL_VERSION 300

#define float3(x, y, z) {{x, y, z}}  // macro to replace ugly initializer braces

#include <iostream>
#include <CL/opencl.hpp>
#include <fstream>
#include "Sphere.hpp"

using namespace std;
using namespace cl;

class OCL {
public:

    OCL(bool info,int w, int h):
            w(w),
            h(h)
    {
        cpu_output.resize(w*h);

        global_work_size = w * h;

        // initialise OpenCL
        initOpenCL(info);

    }
    std::vector<cl_float4> cpu_output;

    void initScenePlanes(int pc){
        this->plane_count = pc;
        cpu_planes.resize(pc);

        cpu_planes[0].position = {-0.35f, -0.0f, -0.3f};
        cpu_planes[0].position2 = {0.30f, -0.0f, 0.3f};
        cpu_planes[0].color = {0.25f, 0.25f, 0.75f};
        cpu_planes[0].emission = {0, 0, 0};
        cpu_planes[0].normal =  {0, 1, 0};
    }

    void initScene(int sc){
        this->sphere_count = sc;
        cpu_spheres.resize(sc);

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

    void setGPUArgs(){
        // Create buffers on the OpenCL device for the image and the scene
        cl_output =     Buffer(context, CL_MEM_WRITE_ONLY, w * h            * sizeof(cl_float3));
        cl_spheres =    Buffer(context, CL_MEM_READ_ONLY, sphere_count   * sizeof(Sphere));
        cl_planes =     Buffer(context, CL_MEM_READ_ONLY, plane_count    * sizeof(Plane));


        queue.enqueueWriteBuffer(cl_spheres, CL_TRUE, 0,  sphere_count * sizeof(Sphere), &cpu_spheres[0]);
        queue.enqueueWriteBuffer(cl_planes,  CL_TRUE, 0,  plane_count * sizeof(Plane),  &cpu_planes[0]);


        // specify OpenCL kernel arguments
        kernel.setArg(0, cl_spheres);
        kernel.setArg(1, cl_planes);
        kernel.setArg(2, w);
        kernel.setArg(3, h);
        kernel.setArg(4, sphere_count);
        kernel.setArg(5, plane_count);
        kernel.setArg(6, samples);
        kernel.setArg(7, bounces);
        kernel.setArg(8, cl_output);

        // every pixel in the image has its own thread or "work item",
        // so the total amount of work items equals the number of pixels

        local_work_size = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

        cout << "Kernel work group size: " << local_work_size << endl;

        // Ensure the global work size is a multiple of local work size
        if (global_work_size % local_work_size != 0)
            global_work_size = (global_work_size / local_work_size + 1) * local_work_size;


    }

    void render(){

        cout << "Rendering started..." << endl;

        // launch the kernel
        queue.enqueueNDRangeKernel(
                kernel,
                0,
                global_work_size,
                local_work_size);

        queue.finish();

        cout << "Rendering done! \nCopying output from GPU device to host" << endl;

        // read and copy OpenCL output to CPU
        queue.enqueueReadBuffer(cl_output, CL_TRUE, 0, w * h * sizeof(cl_float3), &cpu_output[0]);

    }

    void saveImage(){
        // write image to PPM file, a very simple image file format
        // PPM files can be opened with IrfanView (download at www.irfanview.com) or GIMP
        FILE *f = fopen("opencl_raytracer.ppm", "w");
        fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);

        // loop over all pixels, write RGB values
        for (auto& pixel : cpu_output) {
            fprintf(f, "%d %d %d ",
                    toInt(pixel.s[0]),
                    toInt(pixel.s[1]),
                    toInt(pixel.s[2]));
        }
    }


private:
    int plane_count = 0;
    int sphere_count = 0;

    Buffer cl_output;
    Buffer cl_spheres;
    Buffer cl_planes;

    Program program;
    Kernel kernel;
    Platform platform;

    Device device;

    CommandQueue queue;
    Context context;

    std::vector<Sphere> cpu_spheres;
    std::vector<Plane> cpu_planes;

    std::size_t global_work_size;
    std::size_t local_work_size{};

    int samples = 8;
    int bounces = 8;

    const int w;
    const int h;

    inline float clamp(float x){ return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }

    // convert RGB float in range [0,1] to int in range [0, 255] and perform gamma correction
    inline int toInt(float x){ return int( clamp(x)); }


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

        pickPlatform(platforms);

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
        pickDevice( devices);


            cout << "\nUsing OpenCL device: \t" << device.getInfo<CL_DEVICE_NAME>() << endl;
            cout << "\t\t\tMax compute units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
            cout << "\t\t\tMax work group size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;



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

    void pickPlatform(const cl::vector<Platform>& platforms){

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

};



#endif //PROJECT_OCL_H
