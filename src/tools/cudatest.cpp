/*
 * cudatest.cpp
 *
 *  Created on: Jan 25, 2021
 *      Author: teng
 */


#include "../cuda/mygpu.h"
#include "../util/util.h"
#include "../util/query_context.h"
#include "../geometry/geometry.h"

void process_with_gpu(gpu_info *gpu, query_context *ctx);

int main(int argc, char **argv){

	vector<gpu_info *> gpus = get_gpus();

	struct timeval start = get_cur_time();
	query_context ctx;
	int num_grids = 1000000;
	ctx.config.num_objects = 10000000;
	double *data = new double[2*ctx.config.num_objects];
	int *result = new int[num_grids];
	uint *offset_size = new uint[num_grids*2];
	ctx.target[0] = (void *)data;
	ctx.target[1] = (void *)offset_size;
	ctx.target[2] = (void *)result;
	int num_objects_per_grid = ctx.config.num_objects/num_grids;
	int cur_offset = 0;

	for(int i=0;i<num_grids;i++){
		offset_size[i*2] = cur_offset;
		offset_size[i*2+1] = num_objects_per_grid;
//		for(int k=0;k<num_objects_per_grid-1;k++){
//			for(int t = k+1;t<num_objects_per_grid;t++){
//				Point *p1 = (Point *)(data+cur_offset*2+k*2);
//				Point *p2 = (Point *)(data+cur_offset*2+t*2);
//				double dist = p1->distance(*p2,true);
//				if(dist<ctx.config.reach_distance){
//					result[i]++;
//				}
//			}
//		}
		cur_offset += num_objects_per_grid;
	}
	logt("compute with cpu",start);
	offset_size[1] = 100;
	process_with_gpu(gpus[0], &ctx);
	for(gpu_info *g:gpus){
		delete g;
	}
	delete []data;
	delete []result;
	delete []offset_size;

}

