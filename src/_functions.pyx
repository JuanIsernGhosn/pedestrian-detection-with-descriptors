import numpy as np
import cv2 as cv2


def local_binary_pattern(unsigned char[:,:] img):

    cdef int y_lim = img.shape[0]
    cdef int x_lim = img.shape[1]
    cdef double[:,:] texture_map = get_texture_map(img)

    hists = []

    for y in range(0, y_lim - 8, 8):
        for x in range(0, x_lim - 8, 8):
            hist = compute_block_lbp(texture_map, y, x)
            hists.append(hist/np.linalg.norm(hist))

    return np.concatenate(hists)


def uniform_local_binary_pattern(unsigned char[:,:] img, list uniform_patterns):

    cdef int y_lim = img.shape[0]
    cdef int x_lim = img.shape[1]
    cdef double[:,:] texture_map = get_texture_map(img)

    hists = []

    for y in range(0, y_lim - 8, 8):
        for x in range(0, x_lim - 8, 8):
            hist = compute_block_ulbp(texture_map, y, x, uniform_patterns)
            hists.append(hist/np.linalg.norm(hist))

    return np.concatenate(hists)


def get_texture_map(unsigned char[:,:] img):
    cdef int y_lim = img.shape[0]
    cdef int x_lim = img.shape[1]
    cdef double[:,:] texture_map = np.zeros((y_lim, x_lim))
    cdef int val = 0
    cdef int pattern = 0

    for y in range(1, y_lim - 1):
        for x in range(1, x_lim - 1):
            val = img[y, x]
            pattern = 0
            pattern = pattern | (1 << 7) if val <= img[y - 1, x - 1] else pattern
            pattern = pattern | (1 << 6) if val <= img[y - 1, x] else pattern
            pattern = pattern | (1 << 5) if val <= img[y - 1, x + 1] else pattern
            pattern = pattern | (1 << 4) if val <= img[y, x + 1] else pattern
            pattern = pattern | (1 << 3) if val <= img[y + 1, x + 1] else pattern
            pattern = pattern | (1 << 2) if val <= img[y + 1, x] else pattern
            pattern = pattern | (1 << 1) if val <= img[y + 1, x - 1] else pattern
            pattern = pattern | (1 << 0) if val <= img[y, x - 1] else pattern
            texture_map[y, x] = pattern

    texture_map[0, :] = texture_map[1, :]
    texture_map[texture_map.shape[0] - 1, :] = texture_map[texture_map.shape[0] - 2, :]
    texture_map[:, 0] = texture_map[:, 1]
    texture_map[:, texture_map.shape[1] - 1] = texture_map[:, texture_map.shape[1] - 2]

    return texture_map


def compute_block_lbp(double[:,:] texture_map, int i_start, int j_start):

    cdef double[:] hist = np.zeros(256)
    cdef double val = 0

    for i in range(i_start, i_start + 16) :
        for j in range(j_start, j_start + 16) :
            val = texture_map[i,j]
            hist[(<int>val)] = hist[(<int>val)] + 1

    return hist


def compute_block_ulbp(double[:,:] texture_map, int i_start, int j_start, list uniform_patterns):

    cdef double[:] hist = np.zeros(59)
    cdef double val = 0

    for i in range(i_start, i_start + 16) :
        for j in range(j_start, j_start + 16) :
            val = texture_map[i, j]
            if val in uniform_patterns:
                hist[(<int>uniform_patterns.index(val))] = hist[(<int>uniform_patterns.index(val))] + 1
            else:
                hist[len(uniform_patterns)] += 1
    return hist