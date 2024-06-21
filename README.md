### pyopencl_clahe

I implement a pyopencl kernel to process 3D images with **CLAHE** (Contrast Limited Adaptive Histogram Equalization) with GPU acceleration. I compare the performances of the CPU clahe algorithm (skimage.exposure.equalize_adapthist) and my GPU adapted one on 3D organoid microscopy images. I also compare the results to verify that CPU and GPU versions of clahe produce similar equalization. To test it with your own data, add **8-bit** 3D stacks named "*raw.tif" to the /Images folder.
