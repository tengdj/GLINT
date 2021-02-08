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
using namespace std;
typedef unsigned int uint;

#define MAX_DATA_SPACE 10

class gpu_info{
public:
	int device_id;
	size_t mem_size;
	bool busy;
	pthread_mutex_t lock;
	void *d_data[MAX_DATA_SPACE];
	size_t data_size[MAX_DATA_SPACE];


	void init();
	~gpu_info();
	void *get_data(int did, size_t ss);
};

vector<gpu_info *> get_gpus();
void print_gpus();


#endif /* MYGPU_H_ */
