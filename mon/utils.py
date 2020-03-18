import cv2
import numpy as np 

def prep_chip(img_arr, img_size=80):
    """Returns a numpy array

    If the input img is channels first, transposes it to channels last. 
    Reshapes the image chip to the expected size of the input tensor.

    Arguments:
        img_arr: ndarray 
        img_size: image size expected by model
    """
    if (img_arr.shape[0] < img_arr.shape[-1]):
        img_arr = img_arr.transpose((1,2,0))
    img_arr = cv2.resize(img_arr, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    return img_arr


def get_buffer(tree_height, spatial_res=0.1):
    """Returns a float

    Given height of the tree, create a buffer distance as a function of the height.

    Arguments:
        tree_height: float
        spatial_res: float, ensure height and resolution are in the same units
    """
    raw_buff = 0.1 * tree_height + 1
    return round(spatial_res * round(raw_buff/spatial_res),1)
    

def normalize(arr):
    """Returns a numpy array

    Depending on the method of normalization, returns an image array
    normalized. Changing this method allows you to not have to change
    the other scripts. 

    Arguments:
        arr: ndarray
    """
    norm_arr = arr/255.0
    return norm_arr
