/*
 * tracer.cpp
 *
 *  Created on: Jan 20, 2021
 *      Author: teng
 */

#include "tracing.h"

void tracer::process_qtree(){
	struct timeval start = get_cur_time();
	// test contact tracing
	vector<Point *> snapshots;
	vector<vector<Point *>> grids;
	Grid grid(mbr, 10000);
	grids.resize(grid.get_grid_num()+1);
	for(int t=0;t<ctx.duration;t++){
		for(int o=0;o<ctx.num_objects;o++){
			Point *p = trace+t*ctx.num_objects+o;
			snapshots.push_back(p);
			grids[grid.getgrid(p)].push_back(p);
		}
		int counter = 0;
		for(vector<Point *> &ps:grids){
			int len = ps.size();
			for(int i=0;i<len-1;i++){
				for(int j=i+1;j<len;j++){
					ps[i]->distance(*ps[j], true);
					counter++;
				}
			}
			ps.clear();
		}
		//print_points(snapshots);
		snapshots.clear();
		logt("time %d calculate %d",start,t+1, counter);
	}
	grids.clear();
	logt("contact trace",start);
}

void tracer::process(){
	switch(ctx.method){
	case QTREE:
		process_qtree();
		break;
	case GPU:
		break;
	default:
		break;
	}
}


