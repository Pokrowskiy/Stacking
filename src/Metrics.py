import numpy as np
import os
import cv2 as cv
from skimage.metrics import structural_similarity as ssim

import torch
import lpips

loss_fn = lpips.LPIPS(net='alex')

def compute_difference_image(result, original):
    """
    Вычисляет и разницу между двумя изображениями.
    result и original - numpy массивы изображений (BGR или RGB), uint8.
    Возвращает разницу в виде изображения.
    """
    diff = cv.absdiff(result, original)
    if len(diff.shape) == 3:
        gray_diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    else:
        gray_diff = diff
    norm_diff = cv.normalize(diff, None, 0, 255, cv.NORM_MINMAX)
    return norm_diff

def compute_difference_metric(result, original):
    """
    Оценивает различия между двумя изображениями.
    result и original - numpy массивы изображений (BGR или RGB), uint8.
    Возвращает словарь с метриками.
    """

    res_f = result.astype(np.float32)
    orig_f = original.astype(np.float32)

    mae = np.mean(np.abs(res_f - orig_f))
    mse = np.mean((res_f - orig_f) ** 2)
    psnr = cv.PSNR(result, original)


    ssim_index = safe_ssim(result, original)

    def to_tensor_lpips(img):
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        return img_t

    result_t = to_tensor_lpips(result)
    original_t = to_tensor_lpips(original)

    with torch.no_grad():
        lpips_val = loss_fn(result_t, original_t).item()

    return {
        'MAE': mae,
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim_index,
        'LPIPS': lpips_val
    }

def safe_ssim(img1, img2):
    min_size = min(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
    win_size = 7
    if min_size < win_size:
        win_size = min_size if min_size % 2 == 1 else min_size - 1
        if win_size < 3:
            win_size = 3

    return ssim(img1, img2, win_size=win_size, channel_axis=2, data_range=255)