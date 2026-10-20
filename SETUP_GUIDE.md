# 🛠️ Setup & Installation Guide

This guide covers how to set up the environment, compile the CUDA kernels, and run the benchmarks on your local machine.

## 📋 Prerequisites

Before running this project, ensure you have the following hardware and software:

### 1. Hardware
* **NVIDIA GPU:** Required (GTX 10-series or newer recommended).
* **OS:** Windows 10/11 or Linux (Ubuntu 20.04+).
* **Note:** This project *will not work* on AMD GPUs or Apple Silicon (M1/M2/M3) as it relies on proprietary CUDA architecture.

### 2. Software
* **NVIDIA CUDA Toolkit (11.0+):** [Download Here](https://developer.nvidia.com/cuda-downloads).
    * Verify installation by running `nvcc --version` in your terminal.
* **C++ Compiler:**
    * **Windows:** Visual Studio 2019/2022 (Community Edition) with "Desktop development with C++" workload installed.
    * **Linux:** GCC (`sudo apt install build-essential`).
* **Python 3.8+:** (Optional, for generating benchmark graphs and videos).
    * Libraries: `pip install matplotlib seaborn opencv-python numpy`

---

## 📂 Project Structure
Ensure your directory looks like this before compiling:

```text
cuda-convolution/
├── lib/
│   ├── stb_image.h          # Image loading library
│   └── stb_image_write.h    # Image saving library
├── benchmark_plot.py        # Python script to generate graphs
├── check_gpu.cu             # Quick environment test script
├── cuda_blur.cu             # The main GPU Kernels (Naive + Tiled)
├── main.cpp                 # The main CPU baseline implementation
├── input.jpg                # Place your 4K test image here
└── README.md
```

---

## ⚙️ Compilation Instructions

### Step 1: Clone the Repo
```bash
git clone https://github.com/iamartyaa/cuda-convolution.git
cd cuda-convolution
```

### Step 2: Compile the GPU Kernels
We use `nvcc` (NVIDIA Cuda Compiler) to build the executable.

**For Windows (PowerShell/CMD):**
```bash
nvcc cuda_blur.cu -o gpu_blur.exe -O3
```

**For Linux:**
```bash
nvcc cuda_blur.cu -o gpu_blur -O3
```

*Note: The `-O3` flag enables maximum optimization level.*

### Step 3: Compile the CPU Baseline (Optional)
If you want to verify the slow CPU performance yourself:
```bash
nvcc main.cpp -o cpu_blur.exe -O3
```

---

## 🚀 How to Run

1.  **Prepare an Image:**
    * Place a high-resolution image (1920x1080 or 4K) in the root folder.
    * Rename it to `input.jpg`.

2.  **Run the GPU Benchmark:**
    ```bash
    ./gpu_blur.exe
    ```
    * **Expected Output:**
        ```text
        Launching Kernel with Grid: 324x216 blocks
        GPU Time (Naive): 1.791 ms
        GPU Time (Tiled): 1.489 ms ⚡
        ```

3.  **Generate the Graphs (Python):**
    ```bash
    python benchmark_plot.py
    ```
    * This will create `benchmark_final.png` showing the performance comparison.

---

## 🔧 Troubleshooting

**Error: `nvcc fatal : Cannot find compiler 'cl.exe' in PATH`**
* **Cause:** The CUDA compiler cannot find the Visual Studio C++ compiler.
* **Fix:** Add the path to `cl.exe` to your System Environment Variables.
    * Typical Path: `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.xx.xxxx\bin\Hostx64\x64`

**Error: `fatal error: stb_image.h: No such file or directory`**
* **Cause:** The compiler can't find the header files.
* **Fix:** Ensure the `lib/` folder exists and contains the header files. If they are missing, download them from the [stb repository](https://github.com/nothings/stb).

**Error: `unsupported Microsoft Visual Studio version`**
* **Fix:** Run the compilation command with the allow-unsupported flag:
    ```bash
    nvcc cuda_blur.cu -allow-unsupported-compiler -o gpu_blur.exe
    ```