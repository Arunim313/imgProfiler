import numpy as np 
import matplotlib.pyplot as plt 

def read_img(filepath: str, gray: bool = False) -> np.ndarray:
    """
    Read an image file and return it as a NumPy array.
    
    Parameters:
    - filepath (str): Path to the image file.
    - gray (bool, optional): If True, converts the image to grayscale. Defaults to False.
    
    Returns:
    - numpy.ndarray: Image represented as a 2D (grayscale) or 3D (RGB) NumPy array.
    """
    # Read the image using matplotlib
    img = plt.imread(filepath)
    
    # Convert to grayscale if requested
    if gray:
        # Calculate the grayscale representation
        gray_img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        return gray_img
    
    return img

def gray_img(img: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale.

    Parameters:
    - img (np.ndarray): An RGB image as a NumPy array.

    Returns:
    - np.ndarray: The grayscale version of the input image.
    """
    # Use the weighted sum method to convert to grayscale
    grayImage = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]    
    return grayImage


def imshow(img: np.ndarray, title: str = None, cmap: str = None):
    """
    Display an image using matplotlib.

    Parameters:
    - img (np.ndarray): The image to display. Can be grayscale or RGB.
    - title (str, optional): Title for the plot. Defaults to None.
    - cmap (str, optional): Color map for displaying grayscale images. Defaults to None.

    Returns:
    - None
    """
    # Check if the image is grayscale or color
    if len(img.shape) == 2:
        # Grayscale image
        plt.imshow(img, cmap=cmap if cmap else 'gray')
    elif len(img.shape) == 3 and img.shape[2] in [3, 4]:
        # RGB or RGBA image
        plt.imshow(img)
    else:
        raise ValueError("Unsupported image format")

    # Set the title if provided
    if title:
        plt.title(title)

    plt.axis('off')  
    plt.show()
