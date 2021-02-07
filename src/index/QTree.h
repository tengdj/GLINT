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
	double grid_width = 5;// in meters
	double x_buffer = 0;
	double y_buffer = 0;
	bool split_node = true;
	// counter
	int num_leafs = 0;
	QConfig(){}
};

class QTNode{
	double mid_x = 0;
	double mid_y = 0;
	int level = 0;
	QConfig *config = NULL;
	box mbr;
	pthread_mutex_t lk;
	QTNode *children[4] = {NULL,NULL,NULL,NULL};
	vector<Point *> objects;

public:

	QTNode(double low_x, double low_y, double high_x, double high_y, QConfig *conf){
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
		config = conf;
		pthread_mutex_init(&lk, NULL);
	}
	QTNode(box m, QConfig *conf):QTNode(m.low[0], m.low[1], m.high[0], m.high[1],conf){
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

	bool split(){

		bool should_split = config->split_node &&
							objects.size()>=2*config->max_objects &&
				   	   	    level<config->max_level &&
							//config->num_leafs<config->max_leafs &&
							mbr.width(true)>config->grid_width;
		if(!should_split){
			return false;
		}

		children[bottom_left] = new QTNode(mbr.low[0],mbr.low[1],mid_x,mid_y, config);
		children[bottom_right] = new QTNode(mid_x,mbr.low[1],mbr.high[0],mid_y, config);
		children[top_left] = new QTNode(mbr.low[0],mid_y,mid_x,mbr.high[1], config);
		children[top_right] = new QTNode(mid_x,mid_y,mbr.high[0],mbr.high[1], config);

		for(int i=0;i<4;i++){
			children[i]->level = level+1;
		}
		for(Point *p:objects){
			// could be possibly in multiple children
			assert(p);
			assert(config);
			bool top = (p->y>mid_y-config->y_buffer);
			bool bottom = (p->y<mid_y+config->y_buffer);
			bool left = (p->x<mid_x+config->x_buffer);
			bool right = (p->x>mid_x-config->x_buffer);
			if(bottom&&left){
				children[0]->objects.push_back(p);
			}
			if(bottom&&right){
				children[1]->objects.push_back(p);
			}
			if(top&&left){
				children[2]->objects.push_back(p);
			}
			if(top&&right){
				children[3]->objects.push_back(p);
			}
		}
		objects.clear();
		return true;
	}
	inline int which_region(Point *p){
		return 2*(p->y>mid_y)+(p->x>mid_x);
	}
	void insert(Point *p){
		assert(p);
		if(isleaf()){
			lock();
			if(isleaf()){
				objects.push_back(p);
				split();
				unlock();
			}else{
				// if the node is splitted by another thread
				// release the lock and recall the insert function
				unlock();
				insert(p);
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
		if(isleaf()){
			return 1;
		}else{
			int num = 0;
			for(int i=0;i<4;i++){
				num += children[i]->leaf_count();
			}
			return num;
		}
	}
	size_t num_objects(){
		if(isleaf()){
			return objects.size();
		}else{
			size_t num = 0;
			for(int i=0;i<4;i++){
				num += children[i]->num_objects();
			}
			return num;
		}
	}
	void get_leafs(vector<QTNode *> &leafs, bool skip_empty = true){

		if(isleaf()){
			if(!skip_empty||objects.size()>0){
				leafs.push_back(this);
			}
		}else{
			for(int i=0;i<4;i++){
				children[i]->get_leafs(leafs, skip_empty);
			}
		}
	}

	void get_leafs(vector<vector<Point *>> &grids, bool skip_empty = true){

		if(isleaf()){
			if(!skip_empty||objects.size()>0){
				grids.push_back(objects);
			}
		}else{
			for(int i=0;i<4;i++){
				children[i]->get_leafs(grids, skip_empty);
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
	}

	void print(){
		vector<QTNode *> nodes;
		get_leafs(nodes,false);
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

	void lock(){
		pthread_mutex_lock(&lk);
	}
	void unlock(){
		pthread_mutex_unlock(&lk);
	}
};

#endif /* SRC_INDEX_QTREE_H_ */
