clahe_3D_kernel = """
__kernel void clahe_3D(
    __global unsigned char* img,     // Input image, flattened 3D array
    __global unsigned char* result,  // Output image, flattened 3D array
    int width,                       // Width of the image
    int height,                      // Height of the image
    int depth,                       // Depth of the image
    int tileSize,                    // Size of each tile (cube)
    float clipLimit,                 // Clipping limit for histogram equalization
    float minIntensity,              // Minimum intensity value
    float maxIntensity               // Maximum intensity value
    
    
) {
    // Calculate global positions (flattened index) from 3D grid
    int global_x = get_global_id(0);  // X coordinate of the pixel
    int global_y = get_global_id(1);  // Y coordinate of the pixel
    int global_z = get_global_id(2);  // Z coordinate of the pixel

    if (global_x >= width || global_y >= height || global_z >= depth) {
        return; // Out of bounds check
    }

    int numBins = 256; //number of bins (in a 8bit image)
    float hist[256] = {0}; // Histogram array


    // Calculate the start and end coordinates of each tile (current pixel position +/- tilesize/2)
    int x_start = max(0,global_x-tileSize/2);
    int y_start = max(0,global_y-tileSize/2);
    int z_start = max(0,global_z-tileSize/2);
    int x_end = min(x_start + tileSize, width);
    int y_end = min(y_start + tileSize, height);
    int z_end = min(z_start + tileSize, depth);

    // Calculate the histogram for the current tile
    for (int z = z_start; z < z_end; z++) {
        for (int y = y_start; y < y_end; y++) {
            for (int x = x_start; x < x_end; x++) {
                int pixelVal = img[(z * height * width) + (y * width) + x]; // Flattened index calculation
                hist[pixelVal]++;
            }
        }
    }

    // Clip the histogram and redistribute the excess
    float totalPixels = (x_end - x_start) * (y_end - y_start) * (z_end - z_start);
    float limit = clipLimit * totalPixels / numBins;
    float excess = 0.0f;

    for (int i = 0; i < numBins; i++) {
        if (hist[i] > limit) {
            excess += hist[i] - limit;
            hist[i] = limit;
        }
    }

    float increment = excess / numBins;
    for (int i = 0; i < numBins; i++) {
        hist[i] += increment;
    }

    // Compute the cumulative distribution function (CDF)
    float cdf[256];
    cdf[0] = hist[0];
    for (int i = 1; i < numBins; i++) {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    // Normalize the CDF
    for (int i = 0; i < numBins; i++) {
        cdf[i] = (cdf[i] - cdf[0]) / (totalPixels - cdf[0]) * (maxIntensity - minIntensity) + minIntensity;
    }

    // Map the pixel value using the CDF
    int pixelVal = img[(global_z * height * width) + (global_y * width) + global_x]; // Flattened index calculation
    result[(global_z * height * width) + (global_y * width) + global_x] = (unsigned char)cdf[pixelVal];
}
"""