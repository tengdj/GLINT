/*
 * hash.cu
 *
 *  Created on: Apr 23, 2021
 *      Author: teng
 */




#include "stdio.h"
#include "stdint.h"
#include "vector"
#include "hash.h"


#include "unordered_set"
#include "unordered_map"
#include "algorithm"
#include "random"

// 32 bit Murmur3 hash
__device__ uint32_t hash(uint32_t k)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (kHashTableCapacity-1);
}

// Create a hash table. For linear probing, this is just an array of KeyValues
KeyValue* create_hashtable()
{
    // Allocate memory
    KeyValue* hashtable;
    cudaMalloc(&hashtable, sizeof(KeyValue) * kHashTableCapacity);

    // Initialize hash table to empty
    static_assert(kEmpty == 0xffffffff, "memset expected kEmpty=0xffffffff");
    cudaMemset(hashtable, 0xff, sizeof(KeyValue) * kHashTableCapacity);

    return hashtable;
}

// Insert the key/values in kvs into the hashtable
__global__ void gpu_hashtable_insert(KeyValue* hashtable, const KeyValue* kvs, unsigned int numkvs)
{
    unsigned int threadid = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadid < numkvs)
    {
        uint32_t key = kvs[threadid].key;
        uint32_t value = kvs[threadid].value;
        uint32_t slot = hash(key);

        while (true)
        {
            uint32_t prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
            if (prev == kEmpty || prev == key)
            {
                hashtable[slot].value = value;
                return;
            }

            slot = (slot + 1) & (kHashTableCapacity-1);
        }
    }
}

void insert_hashtable(KeyValue* pHashTable, const KeyValue* kvs, uint32_t num_kvs)
{
    // Copy the keyvalues to the GPU
    KeyValue* device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) * num_kvs);
    cudaMemcpy(device_kvs, kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyHostToDevice);

    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_insert, 0, 0);

    // Create events for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Insert all the keys into the hash table
    int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;
    gpu_hashtable_insert<<<gridsize, threadblocksize>>>(pHashTable, device_kvs, (uint32_t)num_kvs);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    printf("    GPU inserted %d items in %f ms (%f million keys/second)\n",
        num_kvs, milliseconds, num_kvs / (double)seconds / 1000000.0f);

    cudaFree(device_kvs);
}

// Lookup keys in the hashtable, and return the values
__global__ void gpu_hashtable_lookup(KeyValue* hashtable, KeyValue* kvs, unsigned int numkvs)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < kHashTableCapacity)
    {
        uint32_t key = kvs[threadid].key;
        uint32_t slot = hash(key);

        while (true)
        {
            if (hashtable[slot].key == key)
            {
                kvs[threadid].value = hashtable[slot].value;
                return;
            }
            if (hashtable[slot].key == kEmpty)
            {
                kvs[threadid].value = kEmpty;
                return;
            }
            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
}

void lookup_hashtable(KeyValue* pHashTable, KeyValue* kvs, uint32_t num_kvs)
{
    // Copy the keyvalues to the GPU
    KeyValue* device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) * num_kvs);
    cudaMemcpy(device_kvs, kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyHostToDevice);

    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_insert, 0, 0);

    // Create events for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Insert all the keys into the hash table
    int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;
    gpu_hashtable_insert << <gridsize, threadblocksize >> > (pHashTable, device_kvs, (uint32_t)num_kvs);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    printf("    GPU lookup %d items in %f ms (%f million keys/second)\n",
        num_kvs, milliseconds, num_kvs / (double)seconds / 1000000.0f);

    cudaFree(device_kvs);
}

// Delete each key in kvs from the hash table, if the key exists
// A deleted key is left in the hash table, but its value is set to kEmpty
// Deleted keys are not reused; once a key is assigned a slot, it never moves
__global__ void gpu_hashtable_delete(KeyValue* hashtable, const KeyValue* kvs, unsigned int numkvs)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < kHashTableCapacity)
    {
        uint32_t key = kvs[threadid].key;
        uint32_t slot = hash(key);

        while (true)
        {
            if (hashtable[slot].key == key)
            {
                hashtable[slot].value = kEmpty;
                return;
            }
            if (hashtable[slot].key == kEmpty)
            {
                return;
            }
            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
}

void delete_hashtable(KeyValue* pHashTable, const KeyValue* kvs, uint32_t num_kvs)
{
    // Copy the keyvalues to the GPU
    KeyValue* device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) * num_kvs);
    cudaMemcpy(device_kvs, kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyHostToDevice);

    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_insert, 0, 0);

    // Create events for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Insert all the keys into the hash table
    int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;
    gpu_hashtable_delete<< <gridsize, threadblocksize >> > (pHashTable, device_kvs, (uint32_t)num_kvs);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    printf("    GPU delete %d items in %f ms (%f million keys/second)\n",
        num_kvs, milliseconds, num_kvs / (double)seconds / 1000000.0f);

    cudaFree(device_kvs);
}

// Iterate over every item in the hashtable; return non-empty key/values
__global__ void gpu_iterate_hashtable(KeyValue* pHashTable, KeyValue* kvs, uint32_t* kvs_size)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < kHashTableCapacity)
    {
        if (pHashTable[threadid].key != kEmpty)
        {
            uint32_t value = pHashTable[threadid].value;
            if (value != kEmpty)
            {
                uint32_t size = atomicAdd(kvs_size, 1);
                kvs[size] = pHashTable[threadid];
            }
        }
    }
}

std::vector<KeyValue> iterate_hashtable(KeyValue* pHashTable)
{
    uint32_t* device_num_kvs;
    cudaMalloc(&device_num_kvs, sizeof(uint32_t));
    cudaMemset(device_num_kvs, 0, sizeof(uint32_t));

    KeyValue* device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) * kNumKeyValues);

    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_iterate_hashtable, 0, 0);

    int gridsize = (kHashTableCapacity + threadblocksize - 1) / threadblocksize;
    gpu_iterate_hashtable<<<gridsize, threadblocksize>>>(pHashTable, device_kvs, device_num_kvs);

    uint32_t num_kvs;
    cudaMemcpy(&num_kvs, device_num_kvs, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<KeyValue> kvs;
    kvs.resize(num_kvs);

    cudaMemcpy(kvs.data(), device_kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyDeviceToHost);

    cudaFree(device_kvs);
    cudaFree(device_num_kvs);

    return kvs;
}

// Free the memory of the hashtable
void destroy_hashtable(KeyValue* pHashTable)
{
    cudaFree(pHashTable);
}


void test_correctness(std::vector<KeyValue> insert_kvs, std::vector<KeyValue> delete_kvs, std::vector<KeyValue> kvs)
{
    printf("Testing that there are no duplicate keys...\n");
    std::unordered_set<uint32_t> unique_keys;
    for (uint32_t i = 0; i < kvs.size(); i++)
    {
        if (i % 10000000 == 0)
            printf("    Verifying %d/%d\n", i, (uint32_t)kvs.size());

        KeyValue* node = &kvs[i];
        if (unique_keys.find(node->key) != unique_keys.end())
        {
            printf("Duplicate key found in GPU hash table at slot %d\n", i);
            exit(-1);
        }
        unique_keys.insert(node->key);
    }

    printf("Building unordered_map from original list...\n");
    std::unordered_map<uint32_t, std::vector<uint32_t>> all_kvs_map;
    for (int i = 0; i < insert_kvs.size(); i++)
    {
        if (i % 10000000 == 0)
            printf("    Inserting %d/%d\n", i, (uint32_t)insert_kvs.size());

        auto iter = all_kvs_map.find(insert_kvs[i].key);
        if (iter == all_kvs_map.end())
        {
            all_kvs_map[insert_kvs[i].key] = std::vector<uint32_t>({ insert_kvs[i].value });
        }
        else
        {
            iter->second.push_back(insert_kvs[i].value);
        }
    }

    for (int i = 0; i < delete_kvs.size(); i++)
    {
        if (i % 10000000 == 0)
            printf("    Deleting %d/%d\n", i, (uint32_t)delete_kvs.size());

        auto iter = all_kvs_map.find(delete_kvs[i].key);
        if (iter != all_kvs_map.end())
        {
            all_kvs_map.erase(iter);
        }
    }

    if (unique_keys.size() != all_kvs_map.size())
    {
        printf("# of unique keys in hashtable is incorrect\n");
        exit(-1);
    }

    printf("Testing that each key/value in hashtable is in the original list...\n");
    for (uint32_t i = 0; i < kvs.size(); i++)
    {
        if (i % 10000000 == 0)
            printf("    Verifying %d/%d\n", i, (uint32_t)kvs.size());

        auto iter = all_kvs_map.find(kvs[i].key);
        if (iter == all_kvs_map.end())
        {
            printf("Hashtable key not found in original list\n");
            exit(-1);
        }

        std::vector<uint32_t>& values = iter->second;
        if (std::find(values.begin(), values.end(), kvs[i].value) == values.end())
        {
            printf("Hashtable value not found in original list\n");
            exit(-1);
        }
    }

    printf("Deleting std::unordered_map and std::unique_set...\n");

    return;
}
