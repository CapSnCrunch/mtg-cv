import numpy as np
from skimage.metrics import structural_similarity as ssim

def meanSquaredError(image1, image2):
    error = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    error /= float(image1.shape[0] * image1.shape[1])
    return error

def structuralSimilarity(image1, image2):
    return ssim(image1, image2, win_size=3, data_range=255, gaussian_weights=True, sigma=1.5)