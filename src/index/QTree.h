/*
 * QTree.h
 *
 *  Created on: Jan 1, 2021
 *      Author: teng
 */

#ifndef SRC_INDEX_QTREE_H_
#define SRC_INDEX_QTREE_H_

#include "../geometry/geometry.h"
#include <float.h>
#include <stack>
using namespace std;


/**
 * children layout
 *
 *     2   3
 *
 *     0   1
 * */
enum QT_Direction{
	bottom_left = 0,
	bottom_right = 1,
	top_left = 2,
	top_right = 3
};
class QTNode;

class QConfig{
public:
	// for regulating the split of nodes
	int max_level = INT_MAX;
	int max_leafs = INT_MAX;
	int max_objects = INT_MAX;
	double reach_distance = 5;// in meters
	double x_buffer = 0;
	double y_buffer = 0;
	// counter
	int num_leafs = 0;
	int num_objects = 0;
};

class QTNode{
	double mid_x = 0;
	double mid_y = 0;
	int level = 0;
	QTNode *children[4] = {NULL,NULL,NULL,NULL};
	QConfig *config = NULL;
	box mbr;
public:
	int max_objects = 100;
	vector<Point *> objects;

	QTNode(double low_x, double low_y, double high_x, double high_y){
		mbr.low[0] = low_x;
		mbr.low[1] = low_y;
		mbr.high[0] = high_x;
		mbr.high[1] = high_y;
		mid_x = (mbr.high[0]+mbr.low[0])/2;
		mid_y = (mbr.high[1]+mbr.low[1])/2;
		assert(mbr.low[0]!=mbr.high[0]);
		assert(mbr.low[1]!=mbr.high[1]);
		assert(mbr.low[0]!=mid_x);
		assert(mbr.low[1]!=mid_y);
	}
	QTNode(box m):QTNode(m.low[0], m.low[1], m.high[0], m.high[1]){
	}
	~QTNode(){
		if(!isleaf()){
			for(int i=0;i<4;i++){
				delete children[i];
			}
		}else{
			objects.clear();
		}
	}
	inline bool isleaf(){
		return children[0]==NULL;
	}

	inline bool should_split(){
		return objects.size()>=2*config->max_objects &&
			   level<config->max_level &&
			   config->num_leafs<config->max_leafs &&
			   mbr.width(true)>config->reach_distance/sqrt(2);
	}
	void split(){

		children[bottom_left] = new QTNode(mbr.low[0],mbr.low[1],mid_x,mid_y);
		children[bottom_right] = new QTNode(mid_x,mbr.low[1],mbr.high[0],mid_y);
		children[top_left] = new QTNode(mbr.low[0],mid_y,mid_x,mbr.high[1]);
		children[top_right] = new QTNode(mid_x,mid_y,mbr.high[0],mbr.high[1]);

		for(int i=0;i<4;i++){
			children[i]->level = level+1;
			children[i]->config = config;
		}
		for(Point *p:objects){
			insert(p);
		}
		config->num_leafs += 3;
		config->num_objects -= objects.size();
		objects.clear();
	}
	inline int which_region(Point *p){
		return 2*(p->y>mid_y)+(p->x>mid_x);
	}
	void insert(Point *p){
		if(isleaf()){
			objects.push_back(p);
			config->num_objects++;
			if(should_split()){
				this->split();
			}
		}else{
			// could be possibly in multiple children
			bool top = (p->y>mid_y-config->y_buffer);
			bool bottom = (p->y<mid_y+config->y_buffer);
			bool left = (p->x<mid_x+config->x_buffer);
			bool right = (p->x>mid_x-config->x_buffer);
			if(bottom&&left){
				children[0]->insert(p);
			}
			if(bottom&&right){
				children[1]->insert(p);
			}
			if(top&&left){
				children[2]->insert(p);
			}
			if(top&&right){
				children[3]->insert(p);
			}
		}
	}

	int leaf_count(){
		return config->num_leafs;
	}
	void get_leafs(vector<QTNode *> &leafs, bool skip_empty = true){

		if(isleaf()){
			if(!skip_empty||!objects.size()==0){
				leafs.push_back(this);
			}
		}else{
			for(int i=0;i<4;i++){
				children[i]->get_leafs(leafs);
			}
		}
	}

	void fix_structure(){
		objects.clear();
		if(!isleaf()){
			for(int i=0;i<4;i++){
				children[i]->fix_structure();
			}
		}
		config->max_objects = INT_MAX;
		config->num_objects = 0;
	}

	void print(){
		vector<QTNode *> nodes;
		get_leafs(nodes);
		printf("MULTIPOLYGON(");
		for(int i=0;i<nodes.size();i++){
			if(i>0){
				printf(",");
			}
			printf("((");
			nodes[i]->mbr.print_vertices();
			printf("))");
		}
		printf(")\n");
		nodes.clear();
	}

	void print_node(){
		printf("level: %d objects: %ld width: %f height: %f",level,objects.size(),mbr.width(true),mbr.height(true));
		mbr.print();
		print_points(objects);
	}

	// set the configuration of a qtree node
	// should be called only for the root
	void set_config(QConfig *conf){
		config = conf;
	}
};

#endif /* SRC_INDEX_QTREE_H_ */
