//
// Created by mateo on 01.11.22..
//


#ifndef PROJECT_OCL_H
#define PROJECT_OCL_H

# define CL_HPP_TARGET_OPENCL_VERSION 300

#include <CL/opencl.hpp>
#include "Sphere.hpp"

using namespace cl;
#include <SFML/Graphics.hpp>
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;


struct Settings{
    int image_width;
    int image_height;
    int max_depth;
    int samples;
};

class OCL {
public:
    OCL(Settings s):
            image_width(s.image_width),
            image_height(s.image_height),
            bounces(s.max_depth),
            samples(s.samples)
    {
        cpu_planes.resize(numPlanes);
        cpu_spheres.resize(numSpheres);

        // allocate memory on CPU to hold the rendered image
        cpu_output.resize(image_width * image_height);

    }


    void load2Gpu() {
        // Create buffers on the OpenCL device for the image and the scene
        cl_output = Buffer(context, CL_MEM_WRITE_ONLY, image_width * image_height * sizeof(cl_float3));

        cl_spheres = Buffer(context, CL_MEM_READ_ONLY, numSpheres * sizeof(Sphere));
        cl_planes = Buffer(context, CL_MEM_READ_ONLY, numPlanes * sizeof(Plane));
        cl_camera = Buffer(context, CL_MEM_READ_ONLY, sizeof(cl_float3));


        queue.enqueueWriteBuffer(cl_spheres, CL_TRUE, 0, numSpheres  * sizeof(Sphere), cpu_spheres.data());
        queue.enqueueWriteBuffer(cl_planes, CL_TRUE, 0, numPlanes * sizeof(Plane), cpu_planes.data());
        queue.enqueueWriteBuffer(cl_camera, CL_TRUE, 0, sizeof(cl_float3), &camera_origin);



        // specify OpenCL kernel arguments
        kernel.setArg(0, cl_spheres);
        kernel.setArg(1, cl_planes);
        kernel.setArg(2, cl_camera);
        kernel.setArg(3, image_width);
        kernel.setArg(4, image_height);
        kernel.setArg(5, numSpheres);
        kernel.setArg(6, numPlanes);
        kernel.setArg(7, samples);
        kernel.setArg(8, bounces);
        kernel.setArg(9, cl_output);

//        kernel.setArg(0, cl_spheres);
//        kernel.setArg(1, cl_planes);
//        kernel.setArg(2, image_width);
//        kernel.setArg(3, image_height);
//        kernel.setArg(4, numSpheres);
//        kernel.setArg(5, numPlanes);
//        kernel.setArg(6, samples);
//        kernel.setArg(7, bounces);
//        kernel.setArg(8, cl_output);
    }

    void animate(sf::Keyboard::Key key, std::string& edge){

//        cl_float3 orientation = cpu_planes[0].normal;
//
//        cl_float3 translation = {0, 0, 0};
//        translation.x = orientation.x * 0.1f;
//        translation.y = orientation.y * 0.1f;
//        translation.z = orientation.z * 0.1f;



        if (key == sf::Keyboard::W) {
            camera_origin.y += 0.1f;
        }
        if (key == sf::Keyboard::A) {
            camera_origin.x -= 0.1f;
        }
        if (key == sf::Keyboard::S) {
            camera_origin.y -= 0.1f;
        }
        if (key == sf::Keyboard::D) {
            camera_origin.x += 0.1f;
        }
        if (key == sf::Keyboard::Q) {
            camera_origin.z -= 0.1f;
        }
        if (key == sf::Keyboard::E) {
            camera_origin.z += 0.1f;
        }




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

    sf::Uint8*  saveToArray(){
        auto* pixels = new sf::Uint8[image_width * image_height * 4];
        int i = 0;
        for (auto& pixel : cpu_output) {
            pixels[i] = toInt(pixel.s[0]);
            pixels[i+1] = toInt(pixel.s[1]);
            pixels[i+2] = toInt(pixel.s[2]);
            pixels[i+3] = 255;
            i+=4;
        }
        return pixels;




    }

    void addPlane(cl_float3 up_left, cl_float2 size, cl_float3 color, cl_float3 normal, cl_float3 emission){
        Plane plane;
        plane.up_left = up_left;
        plane.normal = normal;
        plane.color = color;
        plane.emission = emission;
        if (normal.x != 0) {
            plane.down_right.x = up_left.x;
            plane.down_right.y = up_left.y + size.x;
            plane.down_right.z = up_left.z + size.y;
        }
        else if (normal.y != 0) {
            plane.down_right.x = up_left.x + size.x;
            plane.down_right.y = up_left.y;
            plane.down_right.z = up_left.z + size.y;
        }
        else if (normal.z != 0) {
            plane.down_right.x = up_left.x + size.x;
            plane.down_right.y = up_left.y + size.y;
            plane.down_right.z = up_left.z;
        }


        numPlanes++;
        cpu_planes.push_back(plane);


    }

    void initScenePlanes(){
        cpu_planes.resize(0);

        // right wall
        addPlane(
                {0.5f, -0.5f, -0.5f},
                {2.0f, 2.0f},
                {0.25f, 0.25f, 0.75f},
                {1.0f, 0.0f, 0.0f},
                {0.0f, 0.0f, 0.0f});

        // left wall
        addPlane(
                {-0.5f, -0.5f, -0.5f},
                {2.0f, 2.0f},
                {0.75f, 0.25f, 0.25f},
                {1.0f, 0.0f, 0.0f},
                {0.0f, 0.0f, 0.0f});

        // top wall
        addPlane(
                {-0.5f, 0.4f, -0.5f},
                {1.0f, 1.0f},
                {1, 1,1},
                {0.0f, 1.0f, 0.0f},
                {0.0f, 0.0f, 0.0f});

        // bottom wall
        addPlane(
                {-0.5f, -0.5f, -0.5f},
                {1.0f, 1.0f},
                {1, 1,1},
                {0.0f, 1.0f, 0.0f},
                {0.0f, 0.0f, 0.0f});
//
//        // back wall
        addPlane(
                {-0.5f, -0.5f, -0.5f},
                {1.0f, 1.0f},
                {1, 1,1},
                {0.0f, 0.0f, 1.0f},
                {0.0f, 0.0f, 0.0f});


        // light
        addPlane(
                {-0.2f, 0.39f, -0.1f},
                {0.5f, 0.5f},
                {1, 1,1},
                {0.0f, 1.0f, 0.0f},
                {9.0f, 8.0f, 9.0f});





    }

    void initSceneSpheres(){

        numSpheres += 2;
        cpu_spheres.resize(numSpheres);


        // left sphere
        cpu_spheres[0].radius   = 0.16f;
        cpu_spheres[0].position = {-0.25f, -0.24f, -0.1f};
        cpu_spheres[0].color    = {0.9f, 0.8f, 0.7f};
        cpu_spheres[0].emission = {0.0f, 0.0f, 0.0f};
        cpu_spheres[0].isTransparent = false;

        // right sphere
        cpu_spheres[1].radius   = 0.16f;
        cpu_spheres[1].position = {0.25f, -0.24f, 0.1f};
        cpu_spheres[1].color    = {0.9f, 0.8f, 0.7f};
        cpu_spheres[1].emission = {0.0f, 0.0f, 0.0f};
        cpu_spheres[1].isTransparent = true;



    }

    void render(){
        //cout << "Rendering started..." << endl;

        // launch the kernel
        queue.enqueueNDRangeKernel(
                kernel,
                0,
                global_work_size,
                local_work_size);

        queue.finish();

        //cout << "Rendering done! \nCopying output from GPU device to host" << endl;

        // read and copy OpenCL output to CPU
        queue.enqueueReadBuffer(
                cl_output,
                CL_TRUE,
                0,
                image_width * image_height * sizeof(cl_float3),
                cpu_output.data());
    }

    void setBounces(int bounces){
        this->bounces = bounces;
        //std::cout << "Bounces set to " << bounces << std::flush;

    }
    void setSamples(int samples){
        this->samples = samples;
        //std::cout << "Samples set to " << samples << std::flush;



    }
private:

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

    // convert RGB float in range [0,1] to int in range [0, 255] and perform gamma correction
    inline int toInt(float x){ return int(std::clamp(x, 0.0f, 1.0f) * 255 + .5f); }




    std::vector<Plane> cpu_planes;
    std::vector<Sphere> cpu_spheres;
    std::vector<cl_float4> cpu_output;
    CommandQueue queue;
    Device device;
    Kernel kernel;
    Platform platform;
    Context context;
    Program program;
    Buffer cl_output;
    Buffer cl_spheres;
    Buffer cl_planes;
    Buffer cl_camera;
    cl_float3 camera_origin = {0.0f, 0.0f, 2.0f};

    const int image_width;
    const int image_height;
    int samples;	// number of samples per pixel
    int bounces;	// number of bounces per ray

    std::size_t global_work_size;
    std::size_t local_work_size;

    int numSpheres = 0;
    int numPlanes = 0;

};


#endif //PROJECT_OCL_H
