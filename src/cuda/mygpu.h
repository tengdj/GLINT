/*
 * mygpu.h
 *
 *  Created on: Dec 9, 2019
 *      Author: teng
 */

#ifndef MYGPU_H_
#define MYGPU_H_

#include <pthread.h>
#include <vector>
#include "../util/util.h"
using namespace std;
typedef unsigned int uint;

#define MAX_DATA_SPACE 10

class gpu_info{
public:
	int device_id;
	char name[256];
	int clock_rate = 0;
	int bus_width = 0;
	size_t mem_size;
	bool busy;
	pthread_mutex_t lock;
	void *d_data[MAX_DATA_SPACE];
	size_t data_size[MAX_DATA_SPACE];
	int compute_capability_major = 0;
	int compute_capability_minor = 0;


	void init();
	~gpu_info();
	void *get_data(int did, size_t ss);
	void print();
	uint size_allocated(){
		uint size = 0;
		for(int i=0;i<MAX_DATA_SPACE;i++){
			size += data_size[i];
		}
		return size;
	}
};

vector<gpu_info *> get_gpus();
void print_gpus();


#endif /* MYGPU_H_ */
