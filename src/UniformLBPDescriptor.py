import cv2 as cv2
import numpy as np
import multiprocess
from itertools import product


class UniformLBPDescriptor:

    def __init__(self, num_points = 8, radius = 1):
        self.num_points = num_points
        self.radius = radius
        self.uniform_patterns = self.__get_uniform_patters(2**num_points)

    def compute(self, img):
        grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        indexes = [range(0, grey_img.shape[0] - 8, 8), range(0, grey_img.shape[1] - 8, 8)]
        indexes = list(product(*indexes))

        pool = multiprocess.Pool()
        feats = [pool.apply_async(self.__compute_block, args=(grey_img, i, j,)) for i, j in indexes]
        pool.close()
        pool.join()

        feats = [feats[hh].get() for hh in range(len(feats))]
        return np.concatenate(feats, axis=None)

    def __compute_block(self, img, i_start, j_start):
        hist = np.zeros((self.uniform_patterns.__len__(),), dtype=int)
        indexes = [range(i_start, i_start + 14), range(j_start, j_start + 14)]
        indexes = list(product(*indexes))
        for i, j in indexes:
            val = self.__compute_pixel(img, i, j)
            if val in self.uniform_patterns:
                hist[self.uniform_patterns.index(val)] = hist[self.uniform_patterns.index(val)] + 1
        return hist

    def __compute_pixel(self, img, i_start, j_start):
        px_val = img[i_start+1,j_start+1]
        code = []
        indexes = zip([i_start, i_start, i_start, i_start + 1, i_start + 2, i_start + 2, i_start + 2, i_start + 1],
                      [j_start, j_start + 1, j_start + 2, j_start + 2, j_start + 2, j_start + 1, j_start, j_start])
        for i in range(indexes.__len__()):
            val = 1 if img[indexes[i][0],indexes[i][1]] >= px_val else 0
            code = np.append(code, val)
        return int(code.dot(2**np.arange(code.size)[::-1]))

    def __get_uniform_patters(self, values):
        uniform_patterns = []
        for i in range(values):
            binary = map(int, format(i, '08b'))
            first = prev = tran = 0
            for j in range(binary.__len__()):
                if j == 0:
                    first = binary[j]
                    prev = binary[j]
                tran = tran + 1 if binary[j] != prev else tran
                tran = tran + 1 if (j == (binary.__len__() - 1)) & (binary[j] != first) else tran
                prev = binary[j]
            uniform_patterns.append(i) if tran <= 2 else uniform_patterns
        return uniform_patterns