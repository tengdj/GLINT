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

enum SchemaType{
	INVALID = 0,
	BRANCH = 1,
	LEAF = 2
};

typedef struct QTSchema{
	uint grid_id = 0;
	short level = 0;
	short type = INVALID;
	short overflow_count = 0;
	short underflow_count = 0;
	double mid_x;
	double mid_y;
	box mbr;
	uint children[4];
	int which(Point *p){
		return (p->y>mid_y)*2+(p->x>mid_x);
	}
}QTSchema;


class QTNode{
	pthread_mutex_t lk;
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
	bool isleaf = true;

	QTNode *children[4] = {NULL,NULL,NULL,NULL};

	// the IDs of each point belongs to this node
	uint *objects = NULL;
	int object_index = 0;
	int capacity = 0;

	void set_id(uint &id){
		if(isleaf){
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
		pthread_mutex_init(&lk,NULL);
	}
	QTNode(box m, configuration *conf, Point *ps):QTNode(m.low[0], m.low[1], m.high[0], m.high[1],conf,ps){
	}
	~QTNode(){
		if(!isleaf){
			for(int i=0;i<4;i++){
				delete children[i];
			}
		}else{
			free(objects);
		}
	}
	void lock(){
		pthread_mutex_lock(&lk);
	}
	void unlock(){
		pthread_mutex_unlock(&lk);
	}

	void query(vector<uint> &result, Point *p){
		if(isleaf){
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
				   	   	    mbr.width(true)/2>config->reach_distance;
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
			Point *p = points+objects[i];
			int loc = (p->y>mid_y)*2+(p->x>mid_x);
			children[loc]->objects[children[loc]->object_index++] = objects[i];
		}
		for(int i=0;i<4;i++){
			children[i]->split();
		}
		free(objects);
		object_index = 0;
		// officially becomes a branch node
		isleaf = false;
		return true;
	}

	void insert(uint pid){
		if(!isleaf){
			// no need to lock other nodes
			Point *p = points+pid;
			int loc = (p->y>mid_y)*2+(p->x>mid_x);
			children[loc]->insert(pid);
		}else{
			lock();
			// is splitted by other threads, retry
			if(!isleaf){
				unlock();
				insert(pid);
			}else{
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
				unlock();
			}
		}
	}

	size_t leaf_count(){
		if(isleaf){
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
		if(isleaf){
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
		if(isleaf){
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
		if(isleaf){
			if(!skip_empty||object_index>0){
				leafs.push_back(this);
			}
		}else{
			for(int i=0;i<4;i++){
				children[i]->get_leafs(leafs, skip_empty);
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
		fprintf(stderr,"MULTIPOLYGON(");
		for(int i=0;i<nodes.size();i++){
			if(i>0){
				fprintf(stderr,",");
			}
			fprintf(stderr,"((");
			nodes[i]->mbr.print_vertices();
			fprintf(stderr,"))");
		}
		fprintf(stderr,")\n");
		nodes.clear();
	}

	Point *get_point(uint pid){
		return points+objects[pid];
	}

	QTSchema to_schema(){
		QTSchema s;
		s.mid_x = mid_x;
		s.mid_y = mid_y;
		s.level = level;
		s.grid_id = node_id;
		if(isleaf){
			s.type = LEAF;
		}else{
			s.type = BRANCH;
		}
		s.mbr = mbr;
		return s;
	}

	void create_schema(QTSchema *schema, uint &offset){
		uint curoff = offset++;
		// copy schema data
		schema[curoff] = to_schema();
		if(!isleaf){
			for(int i=0;i<4;i++){
				schema[curoff].children[i] = offset;
				children[i]->create_schema(schema, offset);
			}
		}
	}
};

#endif /* SRC_INDEX_QTREE_H_ */
