import numpy as np
import cv2

def save_img(filename, np_array):
    cv2.imwrite("{}.png".format(filename),np_array)

def resize(img, height=128, width=128):
    return cv2.resize(img, (width, height))

# Process data from [x_inf,x_sup] to [y_inf,y_sup] 
def process(np_array, x_inf=0., x_sup=255., y_inf=-1., y_sup=1.):
    return ((y_sup-y_inf)*(np.float32(np_array)-x_inf)/(x_sup-x_inf)) + y_inf

# Process data from [0,255] to [-1,1] 
def preprocess(np_array):
    if len(np_array.shape) > 2 and np_array.shape[-1] == 4:
        np_array = cv2.cvtColor(np_array, cv2.COLOR_BGRA2BGR)
    return process(np_array)

# Process data from [-1,1] to [0,255] 
def deprocess(np_array):
    return np.int16(process(np_array, x_inf=-1., x_sup=1., y_inf=0., y_sup=255.))