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

class QTNode{
	double mid_x = 0;
	double mid_y = 0;
	int level = 0;
	QTNode *children[4] = {NULL,NULL,NULL,NULL};

public:
	box mbr;
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

	void leaf_count(int &count){
		if(isleaf()){
			count++;
		}else{
			for(int i=0;i<4;i++){
				children[i]->leaf_count(count);
			}
		}
	}
	void split(){
		children[bottom_left] = new QTNode(mbr.low[0],mbr.low[1],mid_x,mid_y);
		children[bottom_right] = new QTNode(mid_x,mbr.low[1],mbr.high[0],mid_y);
		children[top_left] = new QTNode(mbr.low[0],mid_y,mid_x,mbr.high[1]);
		children[top_right] = new QTNode(mid_x,mid_y,mbr.high[0],mbr.high[1]);
		for(int i=0;i<4;i++){
			children[i]->level = level+1;
			children[i]->max_objects = max_objects;
		}
		for(Point *p:objects){
			children[which_region(p)]->objects.push_back(p);
		}
		objects.clear();
	}
	inline int which_region(Point *p){
		return 2*(p->y>mid_y)+(p->x>mid_x);
	}
	void insert(Point *p){
		if(isleaf()){
			objects.push_back(p);
			if(objects.size()>2*max_objects){
				this->split();
			}
		}else{
			children[which_region(p)]->insert(p);
		}
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
		if(isleaf()){
			objects.clear();
		}else{
			for(int i=0;i<4;i++){
				children[i]->fix_structure();
			}
		}
		max_objects = INT_MAX;
	}

};

inline void print_qtnodes(vector<QTNode *> &nodes){
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
}

#endif /* SRC_INDEX_QTREE_H_ */
