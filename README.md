### Motivation
**CLAHE** (Contrast Limited Adaptive Histogram Equalization) is a contrast-enhancing algorithm commonly used in microsocpy image analysis. Since it lacked an implementation for NVIDIA-CUDA, which is the most widely used framework for parallel scientific image computing, I translated the algorithm to be used by NVIDIA GPUs with PyOpenCL

### Execution
I compare the performances of the CPU clahe algorithm () and my GPU adapted one on 3D organoid microscopy images. I also compare the results to verify that CPU and GPU versions of clahe produce similar equalization. To visualize the results, upload an image to the 'Test_image' folder. To compare the results and performance over a large range of images, upload your 3D stacks into the 'Images' folder.
To test the kernel with your own data, download the notebook and kernel, and add **8-bit** 3D stacks named "*raw.tif" to the /Images folder.

### Notes
Existing implementations for image analysis: 
- ImageJ (CPU): https://imagej.net/plugins/clahe
- Python (CPU): skimage.exposure.equalize_adapthist
- Mac Metal (GPU): https://github.com/YuAo/Accelerated-CLAHE
Most of the pyopencl kernel was generated with chatGPT then manually corrected.
