/* OpenCL based simple sphere path tracer by Sam Lapere, 2016*/
/* based on smallpt by Kevin Beason */
/* http://raytracey.blogspot.com */

__constant float EPSILON = 0.00003f; /* required to compensate for limited float precision */
__constant float PI = 3.14159265359f;

typedef struct Ray{
	float3 origin;
	float3 dir;
} Ray;


typedef struct Sphere{
	float radius;
	float3 pos;
	float3 color;
	float3 emission;
    bool isTransparent;
} Sphere;

typedef struct Plane{
	float3 color;
	float3 emission;
	float3 up_left;
	float3 down_right;
    float3 normal;
} Plane;

static float get_random(unsigned int *seed0, unsigned int *seed1) {

	/* hash the seeds using bitwise AND operations and bitshifts */
	*seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);
	*seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

	unsigned int ires = ((*seed0) << 16) + (*seed1);

	/* use union struct to convert int to float */
	union {
		float f;
		unsigned int ui;
	} res;

	res.ui = (ires & 0x007fffff) | 0x40000000;  /* bitwise AND, bitwise OR */
	return (res.f - 2.0f) / 2.0f;
}

Ray createCamRay(const int x_coord, const int y_coord, const int width, const int height, __constant float3* cam_pos){

	float fx = (float)x_coord / (float)width;  /* convert int in range [0 - width] to float in range [0-1] */
	float fy = (float)y_coord / (float)height; /* convert int in range [0 - height] to float in range [0-1] */

	/* calculate aspect ratio */
	float aspect_ratio = (float)(width) / (float)(height);
	float fx2 = (fx - 0.5f) * aspect_ratio;
	float fy2 = fy - 0.5f;

	/* determine up_left of pixel on screen */
	float3 pixel_pos = (float3)(fx2, -fy2, 0.0f);

	/* create camera ray*/
	Ray ray;
	ray.origin = *cam_pos;
	ray.dir = normalize(pixel_pos - ray.origin); /* vector from camera to pixel on screen */

	return ray;
}

				/* (__global Sphere* sphere, const Ray* ray) */
float intersect_sphere(const Sphere* sphere, const Ray* ray) /* version using local copy of sphere */
{
	float3 rayToCenter = sphere->pos - ray->origin;
	float b = dot(rayToCenter, ray->dir);
	float c = dot(rayToCenter, rayToCenter) - sphere->radius*sphere->radius;
	float disc = b * b - c;

	if (disc < 0.0f) return 0.0f;
	else disc = sqrt(disc);

	if ((b - disc) > EPSILON) return b - disc;
	if ((b + disc) > EPSILON) return b + disc;

	return 0.0f;
}



float intersect_plane(const Plane* plane, const Ray* ray)
{

	float3 middle_point = (plane->up_left + plane->down_right) / 2.0f;
	float3 ray_origin = ray->origin;


	float3 normal = plane->normal;

    float3 normal_reversed = -normal;
	/*
    denom is the dot product of the normal and the ray direction
	it tells us how many units of the ray we need to move to get to the plane
	if denom is 0, the ray is parallel to the plane
	if denom is positive, the ray is pointing away from the plane
	if denom is negative, the ray is pointing towards the plane
    */
	float denom = dot(normal, ray->dir);
    float denom_reversed = dot(normal_reversed, ray->dir);

	float t = 0.0f;

	if (denom > 1e-6) {
        /*
        p0l0 is a vector between the ray origin and a point on the plane
        in this case, a point in the middle of the plane
        we use this to calculate the distance from the ray origin to the plane
        by taking the dot product of the normal and p0l0
        we then divide this by the dot product of the normal and the ray direction
        this gives us the distance along the ray that we need to travel to get to the plane
        we then multiply this by the ray direction to get the point on the plane that the ray intersects
        */
		float3 p0l0 = middle_point - ray_origin;
		t = dot(p0l0, normal) / denom;

		float3 intersection = ray_origin + t * ray->dir;

		if (intersection.x >= plane->up_left.x - EPSILON && intersection.x <= plane->down_right.x + EPSILON  &&
			intersection.z >= plane->up_left.z - EPSILON  && intersection.z <= plane->down_right.z  + EPSILON &&
			intersection.y >= plane->up_left.y - EPSILON && intersection.y <= plane->down_right.y + EPSILON) {
			return t;
		}

  }

    if (denom_reversed> 1e-6) {
        /*
        p0l0 is a vector between the ray origin and a point on the plane
        in this case, a point in the middle of the plane
        we use this to calculate the distance from the ray origin to the plane
        by taking the dot product of the normal and p0l0
        we then divide this by the dot product of the normal and the ray direction
        this gives us the distance along the ray that we need to travel to get to the plane
        we then multiply this by the ray direction to get the point on the plane that the ray intersects
        */
        float3 p0l0 = middle_point - ray_origin;
        t = dot(p0l0, normal_reversed) / denom_reversed;

        float3 intersection = ray_origin + t * ray->dir;

        if (intersection.x >= plane->up_left.x - EPSILON && intersection.x <= plane->down_right.x + EPSILON  &&
            intersection.z >= plane->up_left.z - EPSILON  && intersection.z <= plane->down_right.z  + EPSILON &&
            intersection.y >= plane->up_left.y - EPSILON && intersection.y <= plane->down_right.y + EPSILON) {
            return t;
        }

    }

	return 0.0f;
}

bool intersect_scene(
		__constant Sphere* spheres,
		__constant Plane* planes,
 			const Ray* ray,
			float* t,
			int* object_id,
			const int sphere_count,
			const int plane_count)
{
	/* initialise t to a very large number,
	so t will be guaranteed to be smaller
	when a hit with the scene occurs */

	float inf = 1e20f;
	*t = inf;

	/* check if the ray intersects each sphere in the scene */
	for (int i = 0; i < sphere_count; i++)  {

		Sphere sphere = spheres[i]; /* create local copy of sphere */

		/* float hitdistance = intersect_sphere(&spheres[i], ray); */
		float hitdistance = intersect_sphere(&sphere, ray);
		/* keep track of the closest intersection and hitobject found so far */
		if (hitdistance != 0.0f && hitdistance < *t) {
			*t = hitdistance;
			*object_id = i;
		}
	}
	for (int i = 0; i < plane_count; i++)  {

		Plane plane = planes[i];

		float hitdistanceplane = intersect_plane(&plane, ray);

		if (hitdistanceplane > 0.0f && hitdistanceplane < *t) {
			*t = hitdistanceplane;
			*object_id = i + sphere_count;
		}
	}
	return *t < inf; /* true when ray interesects the scene */
}




/* the path tracing function */
/* computes a path (starting from the camera) with a defined number of bounces, accumulates light/color at each bounce */
/* each ray hitting a surface will be reflected in a random direction (by randomly sampling the hemisphere above the hitpoint) */
/* small optimisation: diffuse ray directions are calculated using cosine weighted importance sampling */

float3 trace(
	__constant Sphere* spheres,
	__constant Plane* planes,
	const Ray* camray,
	const int sphere_count,
	const int plane_count,
	const int* seed0,
	const int* seed1,
	const int max_bounces)
	{

	Ray ray = *camray;

	float3 accum_color = (float3)(0.0f, 0.0f, 0.0f);
	float3 mask = (float3)(1.0f, 1.0f, 1.0f);

	for (int bounces = 0; bounces < max_bounces; bounces++){

		float t;   /* distance to intersection */
		int hitobject_id = 0; /* index of intersected sphere */

		/* if ray misses scene, return background colour */
		if (!intersect_scene(spheres,planes, &ray, &t, &hitobject_id, sphere_count,plane_count))
			return accum_color += mask * (float3)(0.15f, 0.15f, 0.25f);


		float3 hitpoint = ray.origin + ray.dir * t;

		Sphere hitsphere;
		Plane hitplane;
		float3 normal;

		if(hitobject_id < sphere_count){
			hitsphere = spheres[hitobject_id];
			normal = normalize(hitpoint - hitsphere.pos);
		}
		else{
			hitplane = planes[hitobject_id - sphere_count];

                normal = (float3)(0.0f, 1.0f, 0.0f);

		}
		/* t now contains the distance to the closest intersection */
		/* hitsphere_id now contains the index of the closest intersected sphere */

		/* else, we've got a hit! Fetch the closest hit sphere */
		

		/* compute the hitpoint using the ray equation */

		/* compute the surface normal and flip it if necessary to face the incoming ray */
		
		float3 normal_facing = dot(normal, ray.dir) < 0.0f ? normal : normal * (-1.0f);

		/* compute two random numbers to pick a random point on the hemisphere above the hitpoint*/
		float rand1 = 2.0f * PI * get_random(seed0, seed1);
		float rand2 = get_random(seed0, seed1);
		float rand2s = sqrt(rand2);

		/* create a local orthogonal coordinate frame centered at the hitpoint */

		/* this frame will be used to calculate the diffuse ray direction */
		/* the frame is created by calculating two vectors in the plane */
		/* and then calculating the cross product of these two vectors */
		/* the first vector is calculated by taking the cross product of the normal and a vector that is not parallel to the normal */
		/* the second vector is calculated by taking the cross product of the first vector and the normal */
		/* the first vector is stored in the u variable, the second vector is stored in the v variable */
		/* the normal is stored in the w variable */
		/* the three vectors in the frame are not normalised, but that's ok */

		float3 w = normal_facing;
		float3 axis = fabs(w.x) > 0.1f ? (float3)(0.0f, 1.0f, 0.0f) : (float3)(1.0f, 0.0f, 0.0f);
		float3 u = normalize(cross(axis, w));
		float3 v = cross(w, u);

		/* use the coordinte frame and random numbers to compute the next ray direction */
		float3 newdir = normalize(u * cos(rand1)*rand2s + v*sin(rand1)*rand2s + w*sqrt(1.0f - rand2)  )  ;
        //we use cosine weighted importance sampling to sample the hemisphere above the hitpoint
        //this is a small optimisation that makes the image look a bit nicer




		/* add a very small offset to the hitpoint to prevent self intersection */
		ray.origin = hitpoint + normal_facing * EPSILON;
		ray.dir = newdir;

		/* add the colour and light contributions to the accumulated colour */
		if(hitobject_id < sphere_count){
            if(hitsphere.isTransparent){
                bool generateRay = get_random(seed0, seed1) < 0.5f;

                if(generateRay){
                    float3 newdir = normalize(u * cos(rand1)*rand2s + v*sin(rand1)*rand2s + w*sqrt(1.0f - rand2));
                    ray.origin = hitpoint + normal_facing * EPSILON;
                    ray.dir = newdir;
                }
                else{
                    ray.origin = hitpoint + normal_facing * EPSILON;
                    ray.dir = newdir;
                }
            }
            else{
                accum_color += mask * hitsphere.emission;
                mask *= hitsphere.color;
            }
		}
		else{
			accum_color += mask * hitplane.emission;
			mask *= hitplane.color;
		}


		/* perform cosine-weighted importance sampling for diffuse surfaces*/
		mask *= dot(newdir, normal_facing);
	}

	return accum_color;
}

__kernel void render_kernel(
	__constant Sphere* spheres,
	__constant Plane* planes,
    __constant float3* cam_pos,
	const int width,
	const int height,
	const int sphere_count,
	const int plane_count,
	const int samples,
	const int bounces,
	__global float3* output
	)
{
	unsigned int work_item_id = get_global_id(0);	/* the unique global id of the work item for the current pixel */
	unsigned int x_coord = work_item_id % width;			/* x-coordinate of the pixel */
	unsigned int y_coord = work_item_id / width;			/* y-coordinate of the pixel */

	/* seeds for random number generator */
	unsigned int seed0 = x_coord;
	unsigned int seed1 = y_coord;

	Ray camray = createCamRay(x_coord, y_coord, width, height, cam_pos);

	/* add the light contribution of each sample and average over all samples*/
	float3 finalcolor = (float3)(0.0f, 0.0f, 0.0f);
	float invSamples = 1.0f / samples;

	for (int i = 0; i < samples; i++)
		finalcolor += trace(
			spheres,
			planes,
			&camray,
			sphere_count,
			plane_count,
			&seed0, &seed1,
			bounces
			) * invSamples;

	/* store the pixelcolour in the output buffer */
	output[work_item_id] = finalcolor;
}
