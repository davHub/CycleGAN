import numpy as np
import cv2
from matplotlib import pyplot as plt

def save_img(filename, np_array):
    """Save an image
    
    Args:
        filename: the file name
        np_array: the image to save
    """
    cv2.imwrite("{}.png".format(filename),np_array)
    
def display(img):
    """Display an image
    
    Args:
        img: the image to display
    """ 
    im = np.uint8(img)
    plt.imshow(im)
    plt.axis('off')
    plt.show()

def resize(img, height=128, width=128):
    """Resize an 
    
    Args:
        img: the image to resize
        height: the final height
        width: the final width
    """
    return cv2.resize(img, (width, height))

def process(np_array, x_inf=0., x_sup=255., y_inf=-1., y_sup=1.):
    """Process data from [x_inf,x_sup] to [y_inf,y_sup]
    
    Args:
        np_array: the array to process
        x_inf: the lower bound of current data
        x_sup: the upper bound of current data
        y_inf: the lower bound of desired data
        y_sup: the upper bound of desired data
    """
    return ((y_sup-y_inf)*(np.float32(np_array)-x_inf)/(x_sup-x_inf)) + y_inf

def preprocess(np_array):
    """Process data from [0,255] to [-1,1] 
    
    Args:
        np_array: the array to process
    """
    return process(np_array)

def deprocess(np_array):
    """Process data from [-1,1] to [0,255]
     
    Args:
        np_array: the array to process
    """
    return np.int16(process(np_array, x_inf=-1., x_sup=1., y_inf=0., y_sup=255.))
