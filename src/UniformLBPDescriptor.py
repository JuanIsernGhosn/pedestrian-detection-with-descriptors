import cv2 as cv2
import _functions
import time
import numpy as np

class UniformLBPDescriptor:

    """Uniform Local Binary Pattern descriptor object class

    Attributes:
        (int[]) uniform_patterns: LBP histogram uniform values. 
    """

    uniform_patterns = []

    def __init__(self):
        """Initialize descriptor.
        
        Initialize descriptor and compute uniform patterns.
        """
        self.uniform_patterns = self.__get_uniform_patterns()

    def compute(self, img):
        """Compute descriptor.
        
        Args:
            (int[][][]) img: An RGB image. 
        Returns:
            float[]: Uniform Local Binary Pattern descriptor.
        """
        grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        feat = _functions.uniform_lbp(grey_img, self.uniform_patterns)
        return feat

    def __get_uniform_patterns(self):
        """Compute uniform patterns.
        
        Compute uniform patterns (Number of transitions <= 2) and update instance values.
        
        45 ->  00101101: 
            -Number of transitions: 6 (Non-uniform)
            
        255 -> 11111110:
            -Number of transitions: 2 (Uniform)
        
        Returns:
            int[]: Uniform patterns for descriptor instance.
        """
        patterns = np.zeros(256)
        count = 0
        for i in range(2**8):
            binary = map(int, format(i, '08b'))
            first = prev = tran = 0
            for j in range(binary.__len__()):
                if j == 0:
                    first = binary[j]
                    prev = binary[j]
                tran = tran + 1 if binary[j] != prev else tran
                tran = tran + 1 if (j == (binary.__len__() - 1)) & (binary[j] != first) else tran
                prev = binary[j]
            if tran <= 2:
                count += 1
                patterns[i] = count
        return patterns