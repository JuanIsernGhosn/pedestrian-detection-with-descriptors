import numpy as np
import cv2 as cv2


cpdef lbp(unsigned char[:,:] img):
    """Local Binary Pattern calculation.

    Args:
        (unsigned char[:,:]) img: Greyscale image.
    Returns:
        (double[:]) histograms: Histograms of image blocks.
    """
    cdef int y_lim = img.shape[0]
    cdef int x_lim = img.shape[1]
    cdef double[:,:] texture_map = get_texture_map(img)

    hists = []

    for y in range(0, y_lim - 8, 8):
        for x in range(0, x_lim - 8, 8):
            hist = compute_block_lbp(texture_map, y, x)
            hists.append(hist/np.linalg.norm(hist))

    return np.concatenate(hists)


cpdef uniform_lbp(unsigned char[:,:] img, double[:] uniform_patterns):
    """Uniform Local Binary Pattern calculation.

    Args:
        (unsigned char[:,:]) img: Greyscale image.
        (list) uniform_patterns: List of considered uniform patterns.
    Returns:
        (double[:]) histograms: Histograms of image blocks.
    """
    cdef int y_lim = img.shape[0]
    cdef int x_lim = img.shape[1]
    cdef double[:,:] texture_map = get_texture_map(img)

    hists = []

    for y in range(0, y_lim - 8, 8):
        for x in range(0, x_lim - 8, 8):
            hist = compute_block_ulbp(texture_map, y, x, uniform_patterns)
            hists.append(hist/np.linalg.norm(hist))

    return np.concatenate(hists)


cpdef get_texture_map(unsigned char[:,:] img):
    """LBP texture map calculation.

    Args:
        (unsigned char[:,:]) img: Greyscale image.
    Returns:
        (double[:,:]) texture_map: Texture map with LBP.
    """
    cdef int y_lim = img.shape[0]
    cdef int x_lim = img.shape[1]
    cdef double[:,:] texture_map = np.zeros((y_lim, x_lim))
    cdef int val = 0
    cdef int pt = 0

    for y in range(1, y_lim - 1):
        for x in range(1, x_lim - 1):
            val = img[y, x]
            pt = 0
            pt = pt | (1 << 7) if val <= img[y - 1, x - 1] else pt
            pt = pt | (1 << 6) if val <= img[y - 1, x] else pt
            pt = pt | (1 << 5) if val <= img[y - 1, x + 1] else pt
            pt = pt | (1 << 4) if val <= img[y, x + 1] else pt
            pt = pt | (1 << 3) if val <= img[y + 1, x + 1] else pt
            pt = pt | (1 << 2) if val <= img[y + 1, x] else pt
            pt = pt | (1 << 1) if val <= img[y + 1, x - 1] else pt
            pt = pt | (1 << 0) if val <= img[y, x - 1] else pt
            texture_map[y, x] = pt

    texture_map[0,:] = texture_map[1, :]
    texture_map[texture_map.shape[0]-1,:] = texture_map[texture_map.shape[0]-2,:]
    texture_map[:,0] = texture_map[:, 1]
    texture_map[:,texture_map.shape[1]-1] = texture_map[:,texture_map.shape[1]-2]

    return texture_map


cpdef compute_block_lbp(double[:,:] texture_map, int i_start, int j_start):
    """Local Binary Pattern image block calculation.

    Args:
        (int) i_start: Block x start index.
        (int) j_start: Block y start index.
        (double[:,:]) texture_map: Texture map with LBP.
    Returns:
        (double[:]) histogram: Histogram for each image block.
    """
    cdef double[:] hist = np.zeros(256)
    cdef double val = 0
    cdef int i
    cdef int j

    for i in range(i_start, i_start + 16) :
        for j in range(j_start, j_start + 16) :
            val = texture_map[i,j]
            hist[(<int>val)] = hist[(<int>val)] + 1

    return hist


cpdef compute_block_ulbp(double[:,:] texture_map, int i_start,
                       int j_start, double[:] uniform_patterns):
    """Uniform Local Binary Pattern image block calculation.

    Args:
        (int) i_start: Block x start index.
        (int) j_start: Block y start index.
        (double[:,:]) texture_map: Texture map with LBP.
    Returns:
        (double[:]) histogram: Histogram for each image block.
    """

    cdef double[:] hist = np.zeros(59)
    cdef int val = 0
    cdef double index = 0.0
    cdef int i
    cdef int j

    for i in range(i_start, i_start + 16) :
        for j in range(j_start, j_start + 16) :
            val = (<int>texture_map[i, j])
            if uniform_patterns[val] > 0:
                index = uniform_patterns[val]-1
                hist[(<int>index)] = hist[(<int>index)] + 1.0
            else:
                hist[58] = hist[58] + 1.0
    return hist