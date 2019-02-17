import cv2 as cv2
import numpy as np
import multiprocessing
from itertools import product


class LBPDescriptor:

    def __init__(self, num_points = 8, radius = 1):
        self.num_points = num_points
        self.radius = radius

    def compute(self, img):
        grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        indexes = [range(0, grey_img.shape[0] - 8, 8), range(0, grey_img.shape[1] - 8, 8)]
        indexes = list(product(*indexes))
        feats = []
        pool = multiprocessing.Pool()
        for i,j in indexes:
            feats = np.append(feats, pool.apply_async(self.__compute_block, args=(grey_img,i,j,)))
        pool.close()
        pool.join()
        return feats

    def __compute_block(self, img, i_start, j_start):
        hist = np.zeros((256,), dtype=int)
        indexes = [range(i_start, i_start + 14), range(j_start, j_start + 14)]
        indexes = list(product(*indexes))
        for i, j in indexes:
            val = self.__compute_pixel(img, i, j)
            hist[val] = hist[val]+1
        return hist

    def __compute_pixel(self, img, i_start, j_start):
        px_val = img[i_start+1,j_start+1]
        code = []
        indexes = zip([i_start, i_start, i_start, i_start + 1, i_start + 2, i_start + 2, i_start + 2, i_start + 1],
                      [j_start, j_start + 1, j_start + 2, j_start + 2, j_start + 2, j_start + 1, j_start, j_start])
        for i, j in indexes:
            val = 1 if img[indexes[i][0], indexes[i][1]] >= px_val else 0
            code = np.append(code, val)
        return int(val.dot(2**np.arange(val.size)[::-1]))