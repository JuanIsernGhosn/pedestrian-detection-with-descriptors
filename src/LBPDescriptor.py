import cv2 as cv2
import time
import _functions

class LBPDescriptor:

    def compute(self, img):
        time1 = time.time()
        grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        feat = _functions.local_binary_pattern(grey_img)
        time2 = time.time()
        print str(time2-time1)
        return feat.astype(int)