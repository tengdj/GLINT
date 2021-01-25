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
	double sample_rate = 1;
	qtree->max_objects = sample_rate*(config.duration*config.num_objects/config.num_grids);
	size_t qcount = 0;
	for(int t=0;t<config.duration;t++){
		for(int o=0;o<config.num_objects;o++){
			if(tryluck(sample_rate)){
				Point *p = trace+t*config.num_objects+o;
				assert(mbr.contain(*p));
				qtree->insert(p);
				qcount++;
			}
		}
	}
	logt("building qtree with %ld points with %d max_objects", start, qcount, qtree->max_objects);
	qtree->fix_structure();

	// test contact tracing
	vector<QTNode *> nodes;
	size_t counter = 0;
	for(int t=0;t<config.duration;t++){
		for(int o=0;o<config.num_objects;o++){
			Point *p = trace+t*config.num_objects+o;
			qtree->insert(p);
		}
		qtree->get_leafs(nodes);
		vector<int> gridcount;
		gridcount.resize(nodes.size());
		int tt = 0;
		for(QTNode *ps:nodes){
			int len = ps->objects.size();
			gridcount[tt++] = len;
			if(len>2){
				for(int i=0;i<len-1;i++){
					for(int j=i+1;j<len;j++){
						ps->objects[i]->distance(*ps->objects[j], true);
						counter++;
					}
				}
			}
		}

		sort(gridcount.begin(),gridcount.end(),greater<int>());
		for(int i=0;i<gridcount.size();i++){
			if(!gridcount[i]){
				break;
			}
			cout<<i<<" "<<gridcount[i]<<endl;
		}
		gridcount.clear();

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
	vector<int> formergrid;
	formergrid.resize(config.num_objects);
	vector<int> gridcount;
	gridcount.resize(grid.get_grid_num()+1);
	for(int t=0;t<config.duration;t++){
		int diff = 0;
		for(int o=0;o<config.num_objects;o++){
			Point *p = trace+t*config.num_objects+o;
			int gid = grid.getgrid(p);
			if(gid!=formergrid[o]){
				diff++;
				formergrid[o] = gid;
			}
			grids[gid].push_back(p);
			gridcount[gid]++;
		}
		sort(gridcount.begin(),gridcount.end(),greater<int>());
		for(int i=0;i<gridcount.size();i++){
			if(!gridcount[i]){
				break;
			}
			cout<<i<<" "<<gridcount[i]<<endl;
		}
		//  cout<<diff<<endl;
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
	wf.write((char *)&config.num_objects, sizeof(config.num_objects));
	wf.write((char *)&config.duration, sizeof(config.duration));
	wf.write((char *)&mbr, sizeof(mbr));
	size_t num_points = config.duration*config.num_objects;
	wf.write((char *)trace, sizeof(Point)*num_points);
	wf.close();
	logt("dumped to %s",start_time,path);
}

void tracer::loadFrom(const char *path) {

	int total_num_objects;
	int total_duration;
	struct timeval start_time = get_cur_time();
	ifstream in(path, ios::in | ios::binary);
	in.read((char *)&total_num_objects, sizeof(total_num_objects));
	in.read((char *)&total_duration, sizeof(total_duration));
	in.read((char *)&mbr, sizeof(mbr));
	assert(config.duration<=total_duration);
	assert(config.num_objects<=total_num_objects);

	trace = (Point *)malloc(config.duration*config.num_objects*sizeof(Point));
	for(int i=0;i<config.duration;i++){
		in.read((char *)(trace+i*total_num_objects), config.num_objects*sizeof(Point));
		if(total_num_objects>config.num_objects){
			in.seekg((total_num_objects-config.num_objects)*sizeof(Point), ios_base::cur);
		}
	}

	in.close();
	logt("loaded %d objects last for %d seconds from %s",start_time, config.num_objects, config.duration, path);
	owned_trace = true;
}
