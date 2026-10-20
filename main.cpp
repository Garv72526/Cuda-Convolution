#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image.h"
#include "lib/stb_image_write.h"

#include <iostream>
#include <vector>
#include <chrono> // For high-precision timing

// The "Naive" CPU Convolution (3x3 Box Blur)
// This is intentionally slow to show the difference later.
void cpu_convolution(unsigned char* in, unsigned char* out, int width, int height, int channels) {
    // Loop over every pixel (Y, X)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            
            // Loop over 3 color channels (RGB)
            for (int c = 0; c < channels; c++) {
                int sum = 0;
                
                // The Filter Loop (3x3 Neighbors)
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        // Calculate neighbor index
                        int pixel_idx = ((y + ky) * width + (x + kx)) * channels + c;
                        sum += in[pixel_idx];
                    }
                }
                
                // Average the 9 pixels (Box Blur)
                int center_idx = (y * width + x) * channels + c;
                out[center_idx] = (unsigned char)(sum / 9);
            }
        }
    }
}

int main() {
    // 1. Load Image
    int width, height, channels;
    unsigned char* img_in = stbi_load("input.jpg", &width, &height, &channels, 0);
    
    if (!img_in) {
        std::cerr << "Error: Could not load input.jpg!" << std::endl;
        return -1;
    }

    std::cout << "Image Loaded: " << width << "x" << height << " (" << channels << " channels)" << std::endl;

    // 2. Allocate Output Memory
    size_t img_size = width * height * channels;
    std::vector<unsigned char> img_out(img_size);

    // 3. Run CPU Benchmark
    std::cout << "Running CPU Convolution... (This might take a while)" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    cpu_convolution(img_in, img_out.data(), width, height, channels);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "CPU Time: " << duration.count() << " ms" << std::endl;

    // 4. Save Result
    stbi_write_jpg("output_cpu.jpg", width, height, channels, img_out.data(), 100);
    std::cout << "Saved output_cpu.jpg!" << std::endl;

    // Cleanup
    stbi_image_free(img_in);
    return 0;
}