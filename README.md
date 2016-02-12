# cuStreamComp
Efficient CUDA Stream Compaction Library

Based on 

1)Markus Billeter et al. Efficient Stream Compaction on Wide SIMD Many-Core Architectures

2)InK-Compact-: In kernel Stream Compaction and Its Application to Multi-kernel Data Visualization on GPGPU- D.M. Hughes

It is CUDA efficient stream compaction procedure based on warp ballotting intrinsic.

Usage is straightforward:

Create a predicate that decide whether an element is valid or not.
```
struct predicate
{
	__host__ __device__
	bool operator()(const int x)
	{
		return x>0;
	}
};
```
...
Call the compact procedure to obtain the compacted array d_output.

d_data, d_output have to be allocated on device.
```
cuCompactor::compact<int>(d_data,d_output,length,predicate(),blockSize);
```

*PERFORMANCE*

![Alt text](/results/res.jpg?raw=true "Thrust Performance Comparison")
