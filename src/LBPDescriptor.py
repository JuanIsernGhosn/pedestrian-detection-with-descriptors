import cv2 as cv2
import time
import _functions

class LBPDescriptor:

    def compute(self, img):
        grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        feat = _functions.local_binary_pattern(grey_img)
        return feat