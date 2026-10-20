# 🎓 Technical Deep Dive: Concepts & Architecture

This document explains the GPU concepts, memory hierarchy, and algorithm design used in this project. It is written for engineers who want to understand the "under the hood" mechanics of CUDA.

---

## 1. The Host (CPU) vs. Device (GPU) Model
CUDA uses a heterogeneous programming model.
* **Host:** The CPU. It acts as the commander. It allocates memory (`cudaMalloc`), moves data (`cudaMemcpy`), and launches kernels.
* **Device:** The GPU. It acts as the worker. It executes thousands of threads in parallel.

**In this project:**
1.  CPU loads the JPEG image into RAM.
2.  CPU copies the raw pixel bytes to the GPU VRAM (Global Memory).
3.  CPU triggers the `kernel<<<Grid, Block>>>`.
4.  GPU processes pixels in parallel.
5.  CPU copies the result back to RAM to save the image.

---

## 2. The Naive Kernel (Global Memory)
*Found in function: `blur_kernel_naive`*

In the naive approach, we map **1 Thread to 1 Pixel**.
Every thread calculates the average of its 3x3 neighborhood.

### The Problem: Memory Bandwidth
For a 3x3 blur, every thread reads 9 pixels.
* Thread (x, y) reads `pixel[x][y]` and its 8 neighbors.
* Thread (x+1, y) reads `pixel[x+1][y]` and *its* 8 neighbors.

**Notice the overlap:** Most pixels are read 9 times by 9 different threads.
Because we are reading from **Global Memory (VRAM)** (which is slow/high latency), the GPU cores spend most of their time *waiting* for data. This is called being **Memory Bound**.

---

## 3. The Optimized Kernel (Shared Memory Tiling)
*Found in function: `blur_kernel_tiled`*

To fix the bandwidth issue, we use **Shared Memory**.
Shared Memory is a user-managed L1 Cache. It is on-chip and extremely fast (almost as fast as registers).

### The Strategy: Tiling
We divide the image into small tiles (e.g., 16x16 pixels).
Instead of each thread fending for itself, the threads in a block cooperate.

1.  **Load Phase:** All 256 threads (16x16) work together to load the corresponding image patch from Global Memory into Shared Memory.
2.  **Sync:** We call `__syncthreads()` to ensure the entire tile is loaded.
3.  **Compute Phase:** Threads perform the 3x3 blur by reading *only* from the fast Shared Memory.

### The Challenge: The Halo (Ghost Cells)
A 16x16 block of output pixels requires an 18x18 block of input pixels (because edge pixels need neighbors outside the block). These extra border pixels are called the **Halo** or **Apron**.

**The Logic:**
Inside the kernel, we check if a thread is sitting at the edge of the tile.
* If `threadIdx.x < RADIUS` (Left Edge), it loads the pixel from the left neighbor block.
* If `threadIdx.x > TILE_SIZE - RADIUS` (Right Edge), it loads the pixel from the right neighbor block.

This ensures that when we compute, we never have to reach back out to slow Global Memory.

---

## 4. CUDA Terminology Used

| Term | Definition | Context in Project |
| :--- | :--- | :--- |
| **Grid** | The entire collection of threads running a kernel. | Covers the full 4K image dimensions. |
| **Block** | A group of threads that execute on the same Streaming Multiprocessor (SM). | We used 16x16 (256 threads) per block. |
| **Warp** | A group of 32 threads that execute instructions in lock-step. | The hardware unit of execution. |
| **Coalescing** | When threads read memory addresses that are consecutive. | We ensured `in[index]` reads were aligned. |
| **Race Condition** | When two threads write to the same memory at once. | Avoided by using a "Gather" pattern (Writing only to own index). |
| **Bank Conflict** | When multiple threads in a warp access the same shared memory bank. | Minimized by padding the shared memory array (often implicit in simple 2D tiles). |

---

## 5. Deep Dive: Memory Coalescing
One of the most critical optimizations in this project is **Coalesced Global Memory Access**.

Imagine VRAM as a warehouse and the threads as workers.
* **Non-Coalesced:** Each worker runs to the warehouse to pick up one distinct box from random shelves. The warehouse truck has to make 32 trips for 32 boxes.
* **Coalesced:** The workers agree to pick up boxes that are sitting right next to each other on the shelf. The warehouse truck can pick up all 32 boxes in a **single transaction**.

**In our Code:**
The formula `int x = blockIdx.x * blockDim.x + threadIdx.x` ensures that confident threads (0, 1, 2...) read consecutive pixels (0, 1, 2...). This perfectly aligns with the GPU's memory bus width (typically 32-bytes or 128-bytes), maximizing throughput.

---

## 6. Visualizing the Index Calculation
The most confusing part of CUDA is calculating the unique ID for each thread.
Code: `int x = blockIdx.x * blockDim.x + threadIdx.x;`

* `threadIdx.x`: My seat number in the specific room (Block).
* `blockIdx.x`: Which room number I am in.
* `blockDim.x`: How many seats are in a room.

This formula converts the local hierarchy into a global array index for the image.