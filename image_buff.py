from collections import deque
import numpy as np
import copy

class ImageBuffer(object):
    def __init__(self,maxlen=50):
        self.maxlen = maxlen
        self.buff = deque(maxlen=maxlen)
                
    def __call__(self, image):
        if self.maxlen <= 1: 
            return image         
        self.buff.append(image)
        if len(self.buff)<self.maxlen:
            return image 
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxlen)
            tmp = copy.copy(self.buff[idx])
            return tmp
        else:
            return image
        
    def __len__(self):
        return len(self.buff)