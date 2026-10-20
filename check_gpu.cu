#include <stdio.h>
#include <cuda_runtime.h>

// Macro to check CUDA errors easily
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void hello_cuda() {
    printf("Hello from GPU Thread %d inside Block %d! 🚀\n", threadIdx.x, blockIdx.x);
}

int main() {
    printf("CPU: Attempting to launch kernel...\n");

    // Launch 1 Block with 5 Threads
    hello_cuda<<<2, 5>>>();

    // 1. Check if the launch itself failed (e.g. invalid arguments)
    gpuErrchk( cudaPeekAtLastError() );

    // 2. Wait for GPU to finish (and check for runtime errors like segfaults)
    gpuErrchk( cudaDeviceSynchronize() );
    
    printf("CPU: The GPU has finished safely.\n");
    return 0;
}