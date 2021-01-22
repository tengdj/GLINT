/*
 * tracer.cpp
 *
 *  Created on: Jan 20, 2021
 *      Author: teng
 */

#include "tracing.h"
#include "../index/QTree.h"

void tracer::process_qtree(){
	struct timeval start = get_cur_time();
	QTNode *qtree = new QTNode(mbr);
	qtree->max_objects = ctx.duration*ctx.num_objects/ctx.num_grids;
	for(int t=0;t<ctx.duration;t++){
		for(int o=0;o<ctx.num_objects;o++){
			Point *p = trace+t*ctx.num_objects+o;
			assert(mbr.contain(*p));
			qtree->insert(p);
		}
	}
	qtree->fix_structure();
	logt("building qtree with %d points", start, ctx.duration*ctx.num_objects);

	// test contact tracing
	vector<QTNode *> nodes;
	int counter = 0;
	for(int t=0;t<ctx.duration;t++){
		for(int o=0;o<ctx.num_objects;o++){
			Point *p = trace+t*ctx.num_objects+o;
			qtree->insert(p);
		}
		qtree->get_leafs(nodes);
		for(QTNode *ps:nodes){
			int len = ps->objects.size();
			if(len>2){
				cout<<len<<endl;
				for(int i=0;i<len-1;i++){
					for(int j=i+1;j<len;j++){
						ps->objects[i]->distance(*ps->objects[j], true);
						counter++;
					}
				}
			}
		}
		nodes.clear();
		qtree->fix_structure();
	}
	delete qtree;
	logt("contact trace with %d calculation use QTree",start,counter);
}

void tracer::process_fixgrid(){
	struct timeval start = get_cur_time();
	// test contact tracing
	int counter = 0;
	vector<vector<Point *>> grids;
	Grid grid(mbr, ctx.num_grids);
	grids.resize(grid.get_grid_num()+1);
	for(int t=0;t<ctx.duration;t++){
		for(int o=0;o<ctx.num_objects;o++){
			Point *p = trace+t*ctx.num_objects+o;
			grids[grid.getgrid(p)].push_back(p);
		}
		for(vector<Point *> &ps:grids){
			int len = ps.size();
			if(len>=2){
				cout<<len<<endl;
				for(int i=0;i<len-1;i++){
					for(int j=i+1;j<len;j++){
						ps[i]->distance(*ps[j], true);
						counter++;
					}
				}
			}
			ps.clear();
		}
	}
	grids.clear();
	logt("contact trace with %d calculation use fixed grid",start,counter);
}

void tracer::process(){
	switch(ctx.method){
	case QTREE:
		process_qtree();
		break;
	case GPU:
		break;
	case FIX_GRID:
		process_fixgrid();
		break;
	default:
		break;
	}
}


