#include <iostream>
#include <assert.h>
#include "cuCompactor.cuh"
#include <chrono>
using namespace std;


struct int_predicate
{
	__host__ __device__
	bool operator()(const int x)
	{
		return x>0;
	}
};

void initiData(int *h_data, uint NELEMENTS,uint &goodElements){
	ushort stride = 4;
	for (int i = 0; i < NELEMENTS; ++i) {
		h_data[i] = i%stride;
		if(h_data[i])
			goodElements++;
	}
}


void printData(int *h_data, uint NELEMENTS){
	for (int i = 0; i < NELEMENTS; ++i) {
		cout<<h_data[i]<<" ";
	}
	cout<<endl;
}


void checkVector(int *h_data,uint NELEMENTS,uint NgoodElements){
	//printf("Checking: %i, %i",NELEMENTS,NgoodElements);
	int_predicate predicate;
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
	int *d_data, *d_output, *h_data;

	//data elements from 2^5 to 2^29
	for(int e=7;e<30;e++){
	//blocksize from 32 to 1024
		for(int b=5;b<=10;b++){

			NELEMENTS=1<<e;
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
			initiData(h_data,NELEMENTS,NgoodElements);

			//printData(h_data,NELEMENTS);

			cudaMemcpy(d_data,h_data,datasize,cudaMemcpyHostToDevice);

			clock_t start = clock();
			cuCompactor::compact<int>(d_data,d_output,NELEMENTS,int_predicate(),blockSize);
			cudaDeviceSynchronize();
			clock_t end = clock();
			unsigned long millis = (end - start) * 1000 / CLOCKS_PER_SEC;


			//copy back results to host
			cudaMemcpy(h_data,d_output,datasize,cudaMemcpyDeviceToHost);
			//printData(h_data,NELEMENTS);
			checkVector(h_data,NELEMENTS,NgoodElements);
			//device memory free
			cudaFree(d_data);
			cudaFree(d_output);
			//host free  memory
			free(h_data);
			printf("(%i,%i,%i)\n",NELEMENTS,blockSize,millis);
		}//for blocksize
	}//for elements
	printf("ALL TEST PASSED");

}
