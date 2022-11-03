#pragma once

#include <CL/opencl.hpp>

class Sphere {
  public:
	cl_float radius;
	cl_float3 position;
	cl_float3 color;
	cl_float3 emission;
    bool isTransparent;
};

class Plane {
	public:
	cl_float3 color;
	cl_float3 emission;
	cl_float3 up_left;
	cl_float3 down_right;
    cl_float3 normal;
};

class Material {
    public:
    cl_float3 color;
    cl_float3 emission;
    bool is_emissive;
    bool is_reflective;
    bool is_refractive;
    cl_float refractive_index;
};