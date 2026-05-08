import numpy as np
import cv2 as cv

def compute_sharpness_abs(image):
    """
    Считает резкость изображения методом абсолютной разницы.

    image: изображение для подсчёта раезкости
    Возвращает карту резкости.
    """
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], dtype=np.float32)
    diff_sum = np.zeros(image.shape[:2], dtype=np.float32)
    for c in range(image.shape[2]):
        channel = image[:,:,c]
        neighbor_sum = cv.filter2D(channel, -1, kernel)
        center = channel
        diff = np.abs(center * 8 - neighbor_sum)
        diff_sum += diff

    sharpness = diff_sum / image.shape[2]
    return sharpness

def compute_sharpness_laplace(image):
    """
    Считает резкость изображения через оператор Лапласа.

    image: изображение для подсчёта раезкости
    Возвращает карту резкости.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    sharpness = np.abs(laplacian)
    return sharpness

def compute_sharpness_laplace_smoothed(image):
    """
    Считает резкость изображения через оператор Лапласа после размытия изображения.

    image: изображение для подсчёта раезкости
    Возвращает карту резкости.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (3, 3), sigmaX=1.0)
    laplacian = cv.Laplacian(blurred, cv.CV_64F)
    sharpness = np.abs(laplacian)
    return sharpness

def sharpness_sobel(image):
    """
    Считает резкость изображения через оператор Собеля.

    image: изображение для подсчёта раезкости
    Возвращает карту резкости.
    """
    channels = cv.split(image)
    grad_x_list = []
    grad_y_list = []
    for ch in channels:
        grad_x = cv.Sobel(ch, cv.CV_64F, 1, 0, ksize=3)
        grad_y = cv.Sobel(ch, cv.CV_64F, 0, 1, ksize=3)
        grad_x_list.append(grad_x)
        grad_y_list.append(grad_y)
    sobel_magnitude = np.hypot(np.sum(grad_x_list, axis=0), np.sum(grad_y_list, axis=0))
    return sobel_magnitude