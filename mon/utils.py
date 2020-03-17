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
    raw_buff = 0.025 * tree_height + 2
    return round(spatial_res * round(raw_buff/spatial_res),1)


def make_square(arr):
    """Returns a square image array
    
    If height is greater than width, find the difference between the two. 
    If the difference is even, add to both sides. 
    If the difference is odd, add extra pixel column to the left.

    If the width is greater than the height, find the difference.
    If the difference is even, add to top and bottom.
    If the differnce is odd, add extra pixel row to the top.

    Arguments:
        arr: ndarray, image to be padded
    """
    if (arr.shape[0] > arr.shape[1]):
        pad = arr.shape[0] - arr.shape[1]
        if (pad % 2 == 0):
            pad_left = int(pad / 2)
            pad_right = int(pad / 2)
        else: 
            pad_left = int((pad + 1) / 2)
            pad_right = pad_left - 1
        img_arr = np.pad(arr, ((0,0), (pad_left, pad_right), (0,0)), mode='constant')
    elif (arr.shape[0] < arr.shape[1]):
        pad = arr.shape[1] - arr.shape[0]
        if (pad % 2 == 0):
            pad_up = int(pad / 2)
            pad_down = int(pad / 2)
        else: 
            pad_up = int((pad + 1) / 2)
            pad_down = pad_up - 1
        img_arr = np.pad(arr, ((pad_up, pad_down), (0,0), (0,0)), mode='constant')
    # Should not get this far, but if it does, just return the array
    else:
        img_arr = arr
    return img_arr

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
