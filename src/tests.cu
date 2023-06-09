#include <iostream>
#include <assert.h>
#include "cuCompactor.cuh"
#include <chrono>
#include <stdlib.h>
using namespace std;

#define MAX_THREADS_PER_GRID (2**31)

#define THREADS_PER_WARP 32
//#define THREADS_PER_BLOCK 1024
//#define WARPS_PER_BLOCK (THREADS_PER_BLOCK/THREADS_PER_WARP)
//#define I_SIZE ((3/2)*THREADS_PER_BLOCK)

struct int_predicate
{
	__host__ __device__
	bool operator()(const int x)
	{
		return x>0;
	}
};
#define randBound (50)//<100
void initiData(int *h_data, uint NELEMENTS,uint &goodElements,bool randomOrStride){
	ushort stride = 4;
	for (int i = 0; i < NELEMENTS; ++i) {
		if(randomOrStride)
			h_data[i] = i%stride;
		else
			h_data[i] =(rand()%100 <= randBound) ? 1 : 0;
		if(h_data[i])
			goodElements++;
	}
	/*
	printf("NELE %u goodElements %u\n",NELEMENTS,goodElements);
	int sum = 0;
	int iter = 0;
	for (int i = 0; i < NELEMENTS; ++i) {
		sum += h_data[i];
		if (i%32 == 0 && i > 0){
			printf("iter %d %d\n",iter++,sum);
			sum = 0;
		}
	}
	printf("\n");
	*/
}


void printData(int *h_data, uint NELEMENTS){
	for (int i = 0; i < NELEMENTS; ++i) {
		cout<<h_data[i]<<" ";
	}
	cout<<endl;
}


void checkVector(int *h_data,uint NELEMENTS,uint NgoodElements){
	//printf("Checking: %i, %i\n",NELEMENTS,NgoodElements);
	int_predicate predicate;
	//for(int i=0;i<NgoodElements;i++){
		//printf("%d pred %d\n",h_data[i],predicate(h_data[i]));
	//}
	for(int i=0;i<NgoodElements;i++){
		assert(predicate(h_data[i]));
	}
	for(int i=NgoodElements;i<NELEMENTS;i++){
		assert(!predicate(h_data[i]));
	}
}



unsigned int NELEMENTS=0;
uint NgoodElements=0;
uint blockSize=8;


int main(){
srand(time(0));
	int *d_data, *d_output, *h_data;
	//data elements from 2^5 to 2^29

	// HYBRID
	for(int e=10;e<21;e++){
	//for(int e=7;e<30;e++){
		//blocksize from 32 to 1024
		// Warp method only handles blockSize 1024
		//for(int b=10;b<=10;b++){
		for(int b=5;b<=10;b++){
			//NELEMENTS=1<<e;
			// Warp method needs inputs of powers of 1024.
			NELEMENTS=(1<<10)<<e;
			NgoodElements=0;
			blockSize=1<<b;
			size_t datasize=sizeof(int)*NELEMENTS;
			//host input/output data
			h_data = (int*) malloc(datasize);
			memset(h_data,0,datasize);
			//device input data
			cudaMalloc(&d_data,datasize);
			//device output data
			cudaMalloc(&d_output,datasize);

			cudaMemset(d_output,0,datasize);
			initiData(h_data,NELEMENTS,NgoodElements,false);

			//printData(h_data,NELEMENTS);

			cudaMemcpy(d_data,h_data,datasize,cudaMemcpyHostToDevice);
			//clock_t start = clock();
			int compact_length = cuCompactor::compact<int>(d_data,d_output,NELEMENTS,int_predicate(),blockSize);
			//cudaDeviceSynchronize();
			//clock_t end = clock();
			//unsigned long millis = (end - start) * 1000 / CLOCKS_PER_SEC;
			assert(compact_length==NgoodElements);
			//copy back results to host
			cudaMemcpy(h_data,d_output,datasize,cudaMemcpyDeviceToHost);
			//printData(h_data,NELEMENTS);
			//checkVector(h_data,NELEMENTS,NgoodElements);
			cudaMemset(d_output,0,datasize);
			compact_length = cuCompactor::compactHybrid<int>(d_data,d_output,NELEMENTS,int_predicate(),blockSize);
			//cudaDeviceSynchronize();
			//clock_t end = clock();
			//unsigned long millis = (end - start) * 1000 / CLOCKS_PER_SEC;
			assert(compact_length==NgoodElements);
			//copy back results to host
			cudaMemcpy(h_data,d_output,datasize,cudaMemcpyDeviceToHost);
			//printData(h_data,NELEMENTS);
			//checkVector(h_data,NELEMENTS,NgoodElements);
			cudaMemset(d_output,0,datasize);
			compact_length = cuCompactor::compactThrust<int>(d_data,d_output,NELEMENTS,int_predicate());
			//cudaDeviceSynchronize();
			//clock_t end = clock();
			//unsigned long millis = (end - start) * 1000 / CLOCKS_PER_SEC;
			assert(compact_length==NgoodElements);
			//copy back results to host
			cudaMemcpy(h_data,d_output,datasize,cudaMemcpyDeviceToHost);
			//printData(h_data,NELEMENTS);
			//checkVector(h_data,NELEMENTS,NgoodElements);


			//device memory free
			cudaFree(d_data);
			cudaFree(d_output);
			//host free  memory
			free(h_data);
			//printf("B,%i,%i,%i\n",NELEMENTS,blockSize,millis);
		}//for blocksize
	}//for elements
	printf("ALL TEST PASSED");
}
