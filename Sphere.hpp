#pragma once

#include <CL/opencl.hpp>

class Sphere {
  public:
	cl_float radius;
	cl_float dummy1;
	cl_float dummy2;
	cl_float dummy3;
	cl_float3 position;
	cl_float3 color;
	cl_float3 emission;
};