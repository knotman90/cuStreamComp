# cuStreamComp
Efficient CUDA Stream Compaction Library

Based on thw folllowing works:

1. Markus Billeter et al. Efficient Stream Compaction on Wide SIMD Many-Core Architectures

2. InK-Compact-: In kernel Stream Compaction and Its Application to Multi-kernel Data Visualization on GPGPU- D.M. Hughes

It is an CUDA efficient implementation of the stream compaction algorithm based on **warp ballotting intrinsic**.

# How to use it
Its usage is straightforward:

 - Create a predicate functor to decide whether an element is valid or not.
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

- Call the compact procedure to obtain the compacted array `d_output`.

```
cuCompactor::compact<int>(d_data,d_output,length,predicate(),blockSize);
```

Note that both the input `d_data` and the output  `d_output` arrays have to be allocated on device.


*PERFORMANCE*

![Alt text](/results/res.jpg?raw=true "Thrust Performance Comparison")
