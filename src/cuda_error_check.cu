#include <stdio.h>
#include <stdlib.h>

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CUDASAFECALL( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDACHECKERROR()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	if ( cudaSuccess != err )
	{
		fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
				file, line, cudaGetErrorString( err ) );

		fprintf( stdout, "cudaSafeCall() failed at %s:%i : %s\n",
						file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
#endif

	return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
				file, line, cudaGetErrorString( err ) );

		fprintf( stdout, "cudaCheckError() failed at %s:%i : %s\n",
						file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if( cudaSuccess != err )
	{
		fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
				file, line, cudaGetErrorString( err ) );

		fprintf( stdout, "cudaCheckError() with sync failed at %s:%i : %s\n",
						file, line, cudaGetErrorString( err ) );

		exit( -1 );
	}
#endif

	return;
}
