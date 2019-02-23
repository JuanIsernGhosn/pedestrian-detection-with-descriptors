import cv2 as cv2
import _functions

class UniformLBPDescriptor:

    uniform_patterns = []

    def __init__(self, pts = 8):
        self.pts = pts
        self.uniform_patterns = self.__get_uniform_patterns(pts)

    def compute(self, img):
        grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        feat = _functions.uniform_local_binary_pattern(grey_img, self.uniform_patterns)
        return feat

    def __get_uniform_patterns(self, pts):
        patterns = []
        for i in range(2**pts):
            binary = map(int, format(i, '08b'))
            first = prev = tran = 0
            for j in range(binary.__len__()):
                if j == 0:
                    first = binary[j]
                    prev = binary[j]
                tran = tran + 1 if binary[j] != prev else tran
                tran = tran + 1 if (j == (binary.__len__() - 1)) & (binary[j] != first) else tran
                prev = binary[j]
            patterns.append(i) if tran <= 2 else self.uniform_patterns

        return patterns