import cv2 as cv2
import _functions

class LBPDescriptor:

    #Local Binary Pattern descriptor object class

    def compute(self, img):
        """
            Compute descriptor.
        Args:
            img (int[][][]): An RGB image. 
        Returns:
            float[]: Local Binary Pattern descriptor.
        """
        grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        feat = _functions.lbp(grey_img)
        return feat