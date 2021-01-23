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
	qtree->max_objects = (config.duration*config.num_objects/config.num_grids)/10;
	size_t qcount = 0;
	for(int t=0;t<config.duration;t++){
		for(int o=0;o<config.num_objects;o++){
			if(tryluck(0.1)){
				Point *p = trace+t*num_objects+o;
				assert(mbr.contain(*p));
				qtree->insert(p);
				qcount++;
			}
		}
	}
	qtree->fix_structure();
	logt("building qtree with %ld points", start, qcount);

	// test contact tracing
	vector<QTNode *> nodes;
	size_t counter = 0;
	for(int t=0;t<config.duration;t++){
		for(int o=0;o<config.num_objects;o++){
			Point *p = trace+t*num_objects+o;
			qtree->insert(p);
		}
		qtree->get_leafs(nodes);
		for(QTNode *ps:nodes){
			int len = ps->objects.size();
			if(len>2){
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
	logt("contact trace with %ld calculation use QTree",start,counter);
}

void tracer::process_fixgrid(){
	struct timeval start = get_cur_time();
	// test contact tracing
	int counter = 0;
	vector<vector<Point *>> grids;
	Grid grid(mbr, config.num_grids);
	log("%f",grid.get_step()*1000);
	grids.resize(grid.get_grid_num()+1);
	for(int t=0;t<config.duration;t++){
		for(int o=0;o<config.num_objects;o++){
			Point *p = trace+t*num_objects+o;
			grids[grid.getgrid(p)].push_back(p);
		}
		for(vector<Point *> &ps:grids){
			int len = ps.size();
			if(len>=2){
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


void tracer::dumpTo(const char *path) {
	struct timeval start_time = get_cur_time();
	ofstream wf(path, ios::out|ios::binary|ios::trunc);
	wf.write((char *)&num_objects, sizeof(num_objects));
	wf.write((char *)&duration, sizeof(duration));
	wf.write((char *)&mbr, sizeof(mbr));
	size_t num_points = duration*num_objects;
	wf.write((char *)trace, sizeof(Point)*num_points);
	wf.close();
	logt("dumped to %s",start_time,path);
}

void tracer::loadFrom(const char *path) {

	struct timeval start_time = get_cur_time();
	ifstream in(path, ios::in | ios::binary);
	in.read((char *)&num_objects, sizeof(num_objects));
	in.read((char *)&duration, sizeof(duration));
	in.read((char *)&mbr, sizeof(mbr));
	size_t num_points = duration*num_objects;
	trace = (Point *)malloc(duration*num_objects*sizeof(Point));
	in.read((char *)trace, sizeof(Point)*num_points);
	in.close();
	logt("loaded %d objects last for %d seconds from %s",start_time, num_objects, duration, path);
	owned_trace = true;
}
