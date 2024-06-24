### pyopenCLAHE

I implement a pyopencl kernel to process 3D micrscopy images with **CLAHE** (Contrast Limited Adaptive Histogram Equalization) with GPU acceleration.
I compare its performances with the CPU clahe implementation from skimage (skimage.exposure.equalize_adapthist). I also compare the results to verify that CPU and GPU versions of clahe produce similar equalization.


### Execution
To test the kernel with your own data, download the notebook and kernel, and add **8-bit** 3D stacks named "*raw.tif" to the /Images folder.
