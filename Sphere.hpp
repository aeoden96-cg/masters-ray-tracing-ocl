#pragma once

#include <CL/opencl.hpp>

class Sphere {
  public:
	cl_float radius;
	cl_float3 position;
	cl_float3 color;
	cl_float3 emission;
};

class Plane {
	public:
	cl_float3 color;
	cl_float3 emission;
	cl_float3 position;
	cl_float3 position2;
	cl_float3 normal;
};