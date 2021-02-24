/*
 * QTree.h
 *
 *  Created on: Jan 1, 2021
 *      Author: teng
 */

#ifndef SRC_INDEX_QTREE_H_
#define SRC_INDEX_QTREE_H_

#include "../geometry/geometry.h"
#include "../util/query_context.h"
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

/*
 *
 * each child is represented as an unsigned integer number
 * the lowest bit is the sign of whether it is a leaf or not
 * the rest bits represent the id if it is a leaf,
 * otherwise point to the offset of the children information
 * of such child
 *
 * */
typedef struct QTSchema{
	uint node_id = 0;
	short level = 0;
	short isleaf = 0;
	double mid_x;
	double mid_y;
	box mbr;
	uint children[4];
}QTSchema;


class QTConfig{
public:
	// for regulating the split of nodes
	int max_level = INT_MAX;
	int max_leafs = INT_MAX;
	int max_objects = INT_MAX;
	// minimum width of each region in meters
	double min_width = 5;
	double x_buffer = 0;
	double y_buffer = 0;
	bool split_node = true;
	// counter
	int num_leafs = 0;

	// the buffer to all the points
	Point *points = NULL;
	QTConfig(){}
};

class QTNode{
public:

	// some globally shared information
	configuration *config = NULL;
	Point *points = NULL;

	// description of current node
	uint node_id = 0;
	short level = 0;
	box mbr;
	double mid_x = 0;
	double mid_y = 0;

	QTNode *children[4] = {NULL,NULL,NULL,NULL};

	// the IDs of each point belongs to this node
	uint *objects = NULL;
	int object_index = 0;
	int capacity = 0;

	void set_id(uint &id){
		if(isleaf()){
			node_id = id++;
		}else{
			for(int i=0;i<4;i++){
				children[i]->set_id(id);
			}
		}
	}

	QTNode(double low_x, double low_y, double high_x, double high_y, configuration *conf, Point *ps){
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
		capacity = conf->grid_capacity;
		objects = (uint *)malloc((capacity)*sizeof(uint));
		points = ps;
	}
	QTNode(box m, configuration *conf, Point *ps):QTNode(m.low[0], m.low[1], m.high[0], m.high[1],conf,ps){
	}
	~QTNode(){
		if(!isleaf()){
			for(int i=0;i<4;i++){
				delete children[i];
			}
		}else{
			free(objects);
		}
	}
	inline bool isleaf(){
		return children[0]==NULL;
	}

	void query(vector<uint> &result, Point *p){
		if(isleaf()){
			result.push_back(this->node_id);
		}else{
			// could be possibly in multiple children with buffers enabled
			bool top = (p->y>mid_y-config->y_buffer);
			bool bottom = (p->y<=mid_y+config->y_buffer);
			bool left = (p->x<=mid_x+config->x_buffer);
			bool right = (p->x>mid_x-config->x_buffer);
			if(bottom&&left){
				children[0]->query(result, p);
			}
			if(bottom&&right){
				children[1]->query(result, p);
			}
			if(top&&left){
				children[2]->query(result, p);
			}
			if(top&&right){
				children[3]->query(result, p);
			}
		}
	}

	bool split(){
		bool should_split = object_index>=config->grid_capacity &&
				   	   	    mbr.width(true)>config->reach_distance;
		if(!should_split){
			return false;
		}

		children[bottom_left] = new QTNode(mbr.low[0],mbr.low[1],mid_x,mid_y, config, points);
		children[bottom_right] = new QTNode(mid_x,mbr.low[1],mbr.high[0],mid_y, config, points);
		children[top_left] = new QTNode(mbr.low[0],mid_y,mid_x,mbr.high[1], config, points);
		children[top_right] = new QTNode(mid_x,mid_y,mbr.high[0],mbr.high[1], config, points);

		for(int i=0;i<4;i++){
			children[i]->level = level+1;
		}
		for(int i=0;i<object_index;i++){
			// reinsert all the objects to next level
			insert(objects[i]);
		}
		free(objects);
		object_index = 0;
		return true;
	}

	void insert(uint pid){
		if(isleaf()){
			// avoid overflow the buffer
			// happens when the grid is too condense
			if(object_index==capacity){
				capacity += config->grid_capacity;
				uint *newobjects = (uint *)malloc(capacity*sizeof(uint));
				memcpy(newobjects,objects,(capacity-config->grid_capacity)*sizeof(uint));
				free(objects);
				objects = newobjects;
			}
			objects[object_index++] = pid;
			split();
		}else{
			// no need to lock other nodes
			Point *p = points+pid;
			// could be possibly in multiple children
			bool top = (p->y>mid_y);
			bool bottom = (p->y<=mid_y);
			bool left = (p->x<=mid_x);
			bool right = (p->x>mid_x);
			if(bottom&&left){
				children[0]->insert(pid);
			}
			if(bottom&&right){
				children[1]->insert(pid);
			}
			if(top&&left){
				children[2]->insert(pid);
			}
			if(top&&right){
				children[3]->insert(pid);
			}
		}
	}

	size_t leaf_count(){
		if(isleaf()){
			return 1;
		}else{
			size_t num = 0;
			for(int i=0;i<4;i++){
				num += children[i]->leaf_count();
			}
			return num;
		}
	}
	size_t node_count(){
		if(isleaf()){
			return 1;
		}else {
			size_t num = 1;
			for(int i=0;i<4;i++){
				num += children[i]->node_count();
			}
			return num;
		}
	}
	size_t num_objects(){
		if(isleaf()){
			return object_index;
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
			if(!skip_empty||object_index>0){
				leafs.push_back(this);
			}
		}else{
			for(int i=0;i<4;i++){
				children[i]->get_leafs(leafs, skip_empty);
			}
		}
	}

	void get_leafs(vector<QTNode *> &grids, vector<size_t> &object_num,bool skip_empty = true){

		if(isleaf()){
			if(!skip_empty||object_index>0){
				grids.push_back(this);
				object_num.push_back(object_index);
			}
		}else{
			for(int i=0;i<4;i++){
				children[i]->get_leafs(grids, object_num, skip_empty);
			}
		}
	}

	void fix_structure(){
		object_index = 0;
		if(!isleaf()){
			for(int i=0;i<4;i++){
				children[i]->fix_structure();
			}
		}
	}

	void finalize(){
		uint id = 0;
		set_id(id);
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

	Point *get_point(uint pid){
		return points+objects[pid];
	}

	QTSchema * create_schema(){
		uint offset = 0;
		QTSchema *schema = new QTSchema[this->node_count()];
		create_schema(schema, offset);
		return schema;
	}
	void create_schema(QTSchema *schema, uint &offset){
		uint curoff = offset++;
		// copy schema data
		schema[curoff].mid_x = mid_x;
		schema[curoff].mid_y = mid_y;
		schema[curoff].level = level;
		schema[curoff].node_id = node_id;
		schema[curoff].isleaf = isleaf();
		schema[curoff].mbr = mbr;
		if(!isleaf()){
			for(int i=0;i<4;i++){
				schema[curoff].children[i] = offset;
				children[i]->create_schema(schema, offset);
			}
		}
	}
};

#endif /* SRC_INDEX_QTREE_H_ */
