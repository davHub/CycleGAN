import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import math

def save_img(filename, np_array):
    """Save an image
    
    Args:
        filename: the file name
        np_array: the image to save
    """
    cv2.imwrite("{}.png".format(filename), np_array)
    
def read_img(filename):
    """Read an image from a filename
    
    Args:
        filename: the file name
        np_array: the image to save
    """
    return mpimg.imread(filename)
    
def group_images(imgs, num_rows=2, num_columns=4):
    """Group several images into one image
    
    Args:
        imgs: images to group
        num_rows: number of rows
        num_columns: number of columns
    """
    number_of_missing_elements = num_columns * num_rows - len(imgs)
    imgs = np.append(imgs,
        np.zeros((number_of_missing_elements, *imgs[0].shape)).astype(imgs.dtype),
        axis=0,
    )
    grid = np.concatenate(
        [
            np.concatenate(
                imgs[index * num_columns : (index + 1) * num_columns], axis=1
            )
            for index in range(num_rows)
        ],
        axis=0,
    )
    return grid

    
def display(img):
    """Display an image
    
    Args:
        img: the image to display
    """ 
    plt.imshow(img)
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
    return np.uint8(process(np_array, x_inf=-1., x_sup=1., y_inf=0., y_sup=255.))

def make_noisy(img, crop_size=256, blur=6, std_bright=.2):
    """Modify data (Data augmentation trick)
     
    Args:
        img: image to process
        crop_size: crop dim
        blur: size of blurred kernel
        std_bright: std deviation of Normal distribution (0 centered) to 
    """
    img_w,img_h = img.shape[:2]
    if img_w>crop_size and img_h>crop_size:
        # Cropping data randomly
        new_h = int(np.ceil(np.random.uniform(1e-2, img_w-crop_size)))
        new_w = int(np.ceil(np.random.uniform(1e-2, img_h-crop_size)))
        new_img = img[new_h:new_h+crop_size, new_w:new_w+crop_size]
    
    # Changing brightness
    bright = np.random.normal(0.,std_bright)    
    new_img = new_img + bright
    new_img[np.where(new_img>1.)] = 1.
    new_img[np.where(new_img<-1.)] = -1.
    
    # Blurring randomly image from triangular distribution
    blurred_size = np.random.randint(np.random.randint(blur)+1)
    if blurred_size>0:
        new_img = cv2.blur(new_img, (blurred_size,blurred_size))
    
    if np.random.random() > 0.5:
        new_img = np.fliplr(new_img)    
    return new_img
