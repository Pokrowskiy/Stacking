import numpy as np
import cv2 as cv
from src.SharpnessFunctions import compute_sharpness_laplace

def combine_images_basic(images_aligned, images_for_analysis):
    """
    images_aligned: чистые выровненные кадры для финальной сборки
    images_for_analysis: денойзенные/размытые кадры для расчета маски

    Возвращает объединённое изображение
    """
    sharpness_maps = np.stack([compute_sharpness_laplace(img) for img in images_for_analysis], axis=0)

    for i in range(len(sharpness_maps)):
        sharpness_maps[i] = cv.GaussianBlur(sharpness_maps[i].astype(np.float32), (7, 7), 0)

    max_indices = np.argmax(sharpness_maps, axis=0)

    images_array = np.array(images_aligned)

    height, width = images_aligned[0].shape[:2]
    yy, xx = np.ogrid[:height, :width]
    result_img = images_array[max_indices, yy, xx]

    return result_img