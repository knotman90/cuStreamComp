
/*
 * cuCompactor.h
 *
 *  Created on: 21/mag/2015
 *      Author: knotman
 */

#ifndef CUCOMPACTOR_H_
#define CUCOMPACTOR_H_

#include <thrust/scan.h>
#include <thrust/host_vector.h>

#include <thrust/device_vector.h>
#include "cuda_error_check.cu"
#define THREADS_PER_WARP 32

namespace cuCompactor {

#define warpSize (32)
#define FULL_MASK 0xffffffff

__host__ __device__ int divup(int x, int y) { return x / y + (x % y ? 1 : 0); }

__device__ __inline__ int pow2i (int e){
	return 1<<e;
}


template <typename T,typename Predicate>
__global__ void computeBlockCounts(T* d_input,int length,int*d_BlockCounts,Predicate predicate){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx < length){
		int pred = predicate(d_input[idx]);
		int BC=__syncthreads_count(pred);

		if(threadIdx.x==0){
			d_BlockCounts[blockIdx.x]=BC; // BC will contain the number of valid elements in all threads of this thread block
		}
	}
}

#define WARP_SZ 32
__device__ inline int lane_id(void) { return threadIdx.x % WARP_SZ; }
template <typename T,typename Predicate>
__global__ void computeWarpCounts(T* d_input,int length,unsigned int *pred,int*d_BlockCounts,Predicate predicate){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (length >> 5) ) // divide by 32
        return;

    int lnid = lane_id();
    int warp_id = tid >> 5; // global warp number
    unsigned int mask;
    int cnt;
    for(int i = 0; i < 32 ; i++) {
        mask = __ballot(predicate(d_input[(warp_id<<10)+(i<<5)+lnid]));
        //mask = __ballot_sync(0xFFFFFFFF,predicate(d_input[(warp_id<<10)+(i<<5)+lnid]));

        if (lnid == 0){
            pred[(warp_id<<5)+i] = mask;
			//printf("pred[%d]=%u\n",(warp_id<<5)+i,mask);
		}
		if (lnid == i){	
		    cnt = __popc(mask);
			//printf("lnid %d cnt %d\n",lnid,cnt);
		}
    }
    // para reduction to a sum of 1024 elements
    #pragma unroll
    for (int offset = 16 ; offset > 0; offset >>= 1)
        cnt += __shfl_down(cnt, offset);
        //cnt += __shfl_down_sync(0xFFFFFFFF,cnt, offset);
		
    if (lnid == 0)
        d_BlockCounts[warp_id] = cnt; // store the sum of the group
}


template <typename T,typename Predicate>
__global__ void compactK(T* d_input,int length, T* d_output,int* d_BlocksOffset,Predicate predicate ){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	extern __shared__ int warpTotals[];
	if(idx < length){
		int pred = predicate(d_input[idx]);
		int w_i = threadIdx.x/warpSize; //warp index
		int w_l = idx % warpSize;//thread index within a warp

		// compute exclusive prefix sum based on predicate validity to get output offset for thread in warp
		int t_m = FULL_MASK >> (warpSize-w_l); //thread mask
		#if (CUDART_VERSION < 9000)
		int b   = __ballot(pred) & t_m; //ballot result = number whose ith bit is one if the ith's thread pred is true masked up to the current index in warp
		#else
		int b	= __ballot_sync(FULL_MASK,pred) & t_m;
		#endif
		int t_u	= __popc(b); // popc count the number of bit one. simply count the number predicated true BEFORE MY INDEX

		// last thread in warp computes total valid counts for the warp
		if(w_l==warpSize-1){
			warpTotals[w_i]=t_u+pred;
		}

		// need all warps in thread block to fill in warpTotals before proceeding
		__syncthreads();

		// first numWarps threads in first warp compute exclusive prefix sum to get output offset for each warp in thread block
		int numWarps = blockDim.x/warpSize;
		unsigned int numWarpsMask = FULL_MASK >> (warpSize-numWarps);
		if(w_i==0 && w_l<numWarps){
			int w_i_u=0;
			for(int j=0;j<=5;j++){ // must include j=5 in loop in case any elements of warpTotals are identically equal to 32
				#if (CUDART_VERSION < 9000)
		                int b_j =__ballot( warpTotals[w_l] & pow2i(j) ); //# of the ones in the j'th digit of the warp offsets
				#else
				int b_j =__ballot_sync(numWarpsMask, warpTotals[w_l] & pow2i(j) );
				#endif
				w_i_u += (__popc(b_j & t_m)  ) << j;
				//printf("indice %i t_m=%i,j=%i,b_j=%i,w_i_u=%i\n",w_l,t_m,j,b_j,w_i_u);
			}
			warpTotals[w_l]=w_i_u;
		}

		// need all warps in thread block to wait until prefix sum is calculated in warpTotals
		__syncthreads(); 

		// if valid element, place the element in proper destination address based on thread offset in warp, warp offset in block, and block offset in grid
		if(pred){
			d_output[t_u+warpTotals[w_i]+d_BlocksOffset[blockIdx.x]]= d_input[idx];
		}


	}
}


template <typename T,typename Predicate>
__global__ void compactKKey(T* d_input,int length, T* d_output,int* d_BlocksOffset,Predicate predicate ){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	extern __shared__ int warpTotals[];
	if(idx < length){
		int pred = predicate(d_input[idx]);
		int w_i = threadIdx.x/warpSize; //warp index
		int w_l = idx % warpSize;//thread index within a warp

		// compute exclusive prefix sum based on predicate validity to get output offset for thread in warp
		int t_m = FULL_MASK >> (warpSize-w_l); //thread mask
		#if (CUDART_VERSION < 9000)
		int b   = __ballot(pred) & t_m; //ballot result = number whose ith bit is one if the ith's thread pred is true masked up to the current index in warp
		#else
		int b	= __ballot_sync(FULL_MASK,pred) & t_m;
		#endif
		int t_u	= __popc(b); // popc count the number of bit one. simply count the number predicated true BEFORE MY INDEX

		// last thread in warp computes total valid counts for the warp
		if(w_l==warpSize-1){
			warpTotals[w_i]=t_u+pred;
		}

		// need all warps in thread block to fill in warpTotals before proceeding
		__syncthreads();

		// first numWarps threads in first warp compute exclusive prefix sum to get output offset for each warp in thread block
		int numWarps = blockDim.x/warpSize;
		unsigned int numWarpsMask = FULL_MASK >> (warpSize-numWarps);
		if(w_i==0 && w_l<numWarps){
			int w_i_u=0;
			for(int j=0;j<=5;j++){ // must include j=5 in loop in case any elements of warpTotals are identically equal to 32
				#if (CUDART_VERSION < 9000)
		                int b_j =__ballot( warpTotals[w_l] & pow2i(j) ); //# of the ones in the j'th digit of the warp offsets
				#else
				int b_j =__ballot_sync(numWarpsMask, warpTotals[w_l] & pow2i(j) );
				#endif
				w_i_u += (__popc(b_j & t_m)  ) << j;
				//printf("indice %i t_m=%i,j=%i,b_j=%i,w_i_u=%i\n",w_l,t_m,j,b_j,w_i_u);
			}
			warpTotals[w_l]=w_i_u;
		}

		// need all warps in thread block to wait until prefix sum is calculated in warpTotals
		__syncthreads(); 

		// if valid element, place the element in proper destination address based on thread offset in warp, warp offset in block, and block offset in grid
		if(pred){
			d_output[t_u+warpTotals[w_i]+d_BlocksOffset[blockIdx.x]]= idx;
		}


	}
}

template <typename T>
__global__ void phase3Key(T* d_input,const int length, T* d_output,int* d_BlockCounts,unsigned int *pred){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (length >> 5) ) // divide by 32
        return;

    int lnid = lane_id();
    int warp_id = tid >> 5; // global warp number

    unsigned int predmask;
    int cnt;

    for(int i = 0; i < 32 ; i++) {
        if (lnid == i) {
            // each thr take turns to load its local var (i.e regs)
            predmask = pred[(warp_id<<5)+i];
            cnt = __popc(predmask);
        }
    }
    // parallel prefix sum

    #pragma unroll
    for (int offset=1; offset<32; offset<<=1) {
        int n = __shfl_up(cnt, offset) ;
        if (lnid >= offset) cnt += n;
    }

    int global_index =0 ;
    if (warp_id > 0)
        global_index = d_BlockCounts[warp_id -1];

    for(int i = 0; i < 32 ; i++) {
        int mask = __shfl(predmask, i); // broadcast from thr i
        int subgroup_index = 0;
        if (i > 0)
        subgroup_index = __shfl(cnt, i-1); // broadcast from thr i-1 if i>0

        if (mask & (1 << lnid ) ) // each thr extracts its pred bit
            d_output[global_index + subgroup_index +
        __popc(mask & ((1 << lnid) - 1))] = (warp_id<<10)+ (i<<5) + lnid;
    }
}

template <typename T>
__global__ void phase3(T* d_input,const int length, T* d_output,int* d_BlockCounts,unsigned int *pred){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (length >> 5) ) // divide by 32
        return;

    int lnid = lane_id();
    int warp_id = tid >> 5; // global warp number

    unsigned int predmask;
    int cnt;

    for(int i = 0; i < 32 ; i++) {
        if (lnid == i) {
            // each thr take turns to load its local var (i.e regs)
            predmask = pred[(warp_id<<5)+i];
            cnt = __popc(predmask);
        }
    }
    // parallel prefix sum

    #pragma unroll
    for (int offset=1; offset<32; offset<<=1) {
        int n = __shfl_up(cnt, offset) ;
        if (lnid >= offset) cnt += n;
    }

    int global_index =0 ;
    if (warp_id > 0)
        global_index = d_BlockCounts[warp_id -1];

    for(int i = 0; i < 32 ; i++) {
        int mask = __shfl(predmask, i); // broadcast from thr i
        int subgroup_index = 0;
        if (i > 0)
        subgroup_index = __shfl(cnt, i-1); // broadcast from thr i-1 if i>0

        if (mask & (1 << lnid ) ) // each thr extracts its pred bit
            d_output[global_index + subgroup_index +
        __popc(mask & ((1 << lnid) - 1))] = d_input[(warp_id<<10)+ (i<<5) + lnid];
    }
}

template <class T>
__global__  void printArray_GPU(T* hd_data, int size,int newline){
	int w=0;
	for(int i=0;i<size;i++){
		if(i%newline==0) {
			printf("\n%i -> ",w);
			w++;
		}
		printf("%i ",hd_data[i]);
	}
	printf("\n");
}

template <typename T,typename Predicate>
int compact(T* d_input,T* d_output,int length, Predicate predicate, int blockSize){
	int numBlocks = divup(length,blockSize);
	int* d_BlocksCount;
	int* d_BlocksOffset;
	CUDASAFECALL (cudaMalloc(&d_BlocksCount,sizeof(int)*numBlocks));
	CUDASAFECALL (cudaMalloc(&d_BlocksOffset,sizeof(int)*numBlocks));
	thrust::device_ptr<int> thrustPrt_bCount(d_BlocksCount);
	thrust::device_ptr<int> thrustPrt_bOffset(d_BlocksOffset);

	//phase 1: count number of valid elements in each thread block
	cudaEvent_t start, stop;
	float millis;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	computeBlockCounts<<<numBlocks,blockSize>>>(d_input,length,d_BlocksCount,predicate);
	
	//phase 2: compute exclusive prefix sum of valid block counts to get output offset for each thread block in grid
	thrust::exclusive_scan(thrustPrt_bCount, thrustPrt_bCount + numBlocks, thrustPrt_bOffset);
	
	//phase 3: compute output offset for each thread in warp and each warp in thread block, then output valid elements
	//compactK<<<numBlocks,blockSize,sizeof(int)*(blockSize/warpSize)>>>(d_input,length,d_output,d_BlocksOffset,predicate);
	compactKKey<<<numBlocks,blockSize,sizeof(int)*(blockSize/warpSize)>>>(d_input,length,d_output,d_BlocksOffset,predicate);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millis,start,stop);
	// end time here
	printf("B,%i,%i,%f\n",length,blockSize,millis);
	// determine number of elements in the compacted list
	int compact_length = thrustPrt_bOffset[numBlocks-1] + thrustPrt_bCount[numBlocks-1];

	cudaFree(d_BlocksCount);
	cudaFree(d_BlocksOffset);

	return compact_length;
}

template <typename T,typename Predicate>
int compactHybrid(T* d_input,T* d_output,int length, Predicate predicate, int blockSize){
	int WARPS_PER_BLOCK = divup(blockSize,THREADS_PER_WARP);

	int numWarps = divup(length,THREADS_PER_WARP*THREADS_PER_WARP);
	int numBlocks = divup(numWarps,WARPS_PER_BLOCK);
	/*
	printf("WARPS_PER_BLOCK %d\n",WARPS_PER_BLOCK);
	printf("numElements %d\n",length);
	printf("numBlocks %d\n",numBlocks);
	printf("numWarps %d\n",numWarps);
	*/
	int* d_BlockCounts;
	int* d_BlocksOffset;
	unsigned int* d_Pred;
	CUDASAFECALL (cudaMalloc(&d_BlockCounts,sizeof(int)*numWarps));
	CUDASAFECALL (cudaMalloc(&d_BlocksOffset,sizeof(int)*numWarps));
	// Each iteration (32) has a pred.
	CUDASAFECALL (cudaMalloc(&d_Pred,sizeof(unsigned int)*numWarps*THREADS_PER_WARP));
	//CUDASAFECALL (cudaMalloc(&d_Pred,sizeof(unsigned int)*numWarps*WARPS_PER_BLOCK));
	//CUDASAFECALL (cudaMalloc(&d_Pred,sizeof(unsigned int)*length));

	thrust::device_ptr<int> thrustPrt_wCount(d_BlockCounts);
	thrust::device_ptr<int> thrustPrt_wOffset(d_BlocksOffset);

	// Start time here
	cudaEvent_t start, stop;
	float millis;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	//phase 1: count number of valid elements in each thread block
	computeWarpCounts<<<numBlocks,blockSize>>>(d_input,length,d_Pred,d_BlockCounts,predicate);

	//phase 2: compute exclusive prefix sum of valid block counts to get output offset for each thread block in grid
	thrust::inclusive_scan(thrustPrt_wCount, thrustPrt_wCount + numWarps, thrustPrt_wOffset);
	
	/*
	thrust::device_ptr<unsigned int> thrustPrt_wPred(d_Pred);
	thrust::host_vector<int> thrustVec_wCount(numWarps);
	thrust::host_vector<int> thrustVec_wOffset(numWarps);
	thrust::device_vector<int> thrustVec_wCount_d(thrustPrt_wCount, thrustPrt_wCount + numBlocks); 
	thrust::device_vector<int> thrustVec_wOffset_d(thrustPrt_wOffset, thrustPrt_wOffset + numBlocks); 
	thrustVec_wCount=thrustVec_wCount_d;
	thrustVec_wOffset=thrustVec_wOffset_d;
	for (auto a : thrustVec_wCount )
		printf("%d ",a);
	printf("\n");
	for (auto a : thrustVec_wOffset )
		printf("%d ",a);
	printf("\n");
	*/
	//phase 3: compute output offset for each thread in warp and each warp in thread block, then output valid elements
	phase3Key<<<numBlocks,blockSize>>>(d_input,length,d_output,d_BlocksOffset,d_Pred);
	//phase3<<<numBlocks,blockSize>>>(d_input,length,d_output,d_BlocksOffset,d_Pred);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millis,start,stop);
	// end time here
	printf("H,%i,%i,%f\n",length,blockSize,millis);
	// determine number of elements in the compacted list
	int compact_length = thrustPrt_wOffset[numWarps-1];
	cudaFree(d_BlockCounts);
	cudaFree(d_BlocksOffset);
	cudaFree(d_Pred);

	return compact_length;
}



template <typename T,typename Predicate>
int compactThrust(T* d_input,T* d_output,int length, Predicate predicate){
	thrust::device_ptr<int> thrustPrt_input(d_input);
	thrust::device_ptr<int> thrustPrt_output(d_output);
	thrust::device_vector<int> thrustVec_input(thrustPrt_input, thrustPrt_input + length); 
	thrust::device_vector<int> thrustVec_output(thrustPrt_output, thrustPrt_output + length);
    typedef thrust::device_vector<int>::iterator IndexIterator;
	// Start time here
	cudaEvent_t start, stop;
	float millis;
	int compact_length = 0;
	try {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		IndexIterator indices_end = thrust::copy_if(thrust::make_counting_iterator(0),
													thrust::make_counting_iterator(length),
													thrustVec_input.begin(),
													thrustVec_output.begin(),
													predicate);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&millis,start,stop);
		// end time here
		printf("T,%i,%f\n",length,millis);
		// determine number of elements in the compacted list
		compact_length = (indices_end-thrustVec_output.begin());
	} catch (const char* msg) {
		std::cerr << msg << std::endl;
	}


	return compact_length;
}



} /* namespace cuCompactor */
#endif /* CUCOMPACTOR_H_ */
