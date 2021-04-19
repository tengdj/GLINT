/*
 * combiner.cpp
 *
 *  Created on: Mar 8, 2021
 *      Author: teng
 */

#include "../util/util.h"
#include "../geometry/geometry.h"
#include <fstream>
#include <omp.h>

int main(int argc, char **argv){

	bool adjust = false;
	box mbr_adj;
	if(argc>2){
		adjust = true;
		ifstream in(argv[2], ios::in | ios::binary);
		if(!in.is_open()){
			log("%s cannot be opened",argv[2]);
			exit(0);
		}
		int fk = 0;
		in.read((char *)&fk, sizeof(fk));
		in.read((char *)&fk, sizeof(fk));
		in.read((char *)&mbr_adj, sizeof(mbr_adj));
		in.close();
	}

	int total_num_objects = 0;
	int total_duration = 0;
	box global_mbr;
	vector<string> files;
	list_files(argv[1], files);
	ifstream **ins = new ifstream*[files.size()];
	int *num_objects = new int[files.size()];
	int max_objects = 0;
	for(int i=0;i<files.size();i++){
		string f = files[i];
		ins[i] = new ifstream(f.c_str(), ios::in | ios::binary);
		if(!ins[i]->is_open()){
			log("%s cannot be opened",f.c_str());
			exit(0);
		}
		int duration;
		box mbr;
		ins[i]->read((char *)&num_objects[i], sizeof(total_num_objects));
		ins[i]->read((char *)&duration, sizeof(total_duration));
		total_num_objects += num_objects[i];
		if(total_duration==0){
			total_duration = duration;
		}
		assert(total_duration==duration);
		ins[i]->read((char *)&mbr, sizeof(mbr));
		global_mbr.update(mbr);
		if(max_objects<num_objects[i]){
			max_objects = num_objects[i];
		}
	}
	cout<<total_num_objects<<" "<<total_duration<<endl;
	global_mbr.print();
	char path[256];
	sprintf(path,"combined.tr");

	ofstream out(path);
	if(!out.is_open()){
		log("%s cannot be opened", path);
		exit(0);
	}
	Point *points = new Point[max_objects];
	out.write((char *)&total_num_objects, sizeof(total_num_objects));
	out.write((char *)&total_duration, sizeof(total_duration));
	if(adjust){
		out.write((char *)&mbr_adj, sizeof(mbr_adj));
	}else{
		out.write((char *)&global_mbr, sizeof(global_mbr));
	}
	for(int i=0;i<total_duration;i++){
		for(int j=0;j<files.size();j++){
			ins[j]->read((char *)points,sizeof(Point)*num_objects[j]);
			if(adjust){
#pragma omp parallel for
				for(int k=0;k<num_objects[j];k++){
					Point *p = points+k;
					p->x = ((p->x-global_mbr.low[0])/(global_mbr.high[0]-global_mbr.low[0]))*(mbr_adj.high[0]-mbr_adj.low[0])+mbr_adj.low[0];
					p->y = ((p->y-global_mbr.low[1])/(global_mbr.high[1]-global_mbr.low[1]))*(mbr_adj.high[1]-mbr_adj.low[1])+mbr_adj.low[1];
				}
			}
			out.write((char *)points,sizeof(Point)*num_objects[j]);
		}
	}
	for(int j=0;j<files.size();j++){
		ins[j]->close();
		delete ins[j];
	}
	out.close();
	delete []points;
	delete []ins;
}
