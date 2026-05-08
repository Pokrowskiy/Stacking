import numpy as np
import cv2 as cv

def build_gaussian_pyramid(image, levels):
    """Строит пирамиду Гаусса."""
    pyramid = [image.astype(np.float32)]
    for i in range(1, levels):
        prev_level = pyramid[i - 1]
        next_level = cv.pyrDown(prev_level)
        pyramid.append(next_level)
    return pyramid

def compute_sharpness_for_pyramid(image):
    """
    Специальная версия compute_sharpness_laplace() расчета резкости для уровней пирамиды.
    """
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv.Laplacian(gray, cv.CV_32F)
    return np.abs(laplacian)

def build_laplacian_pyramid(image, levels):
    """Строит пирамиду Лапласа"""
    gaussian_pyramid = build_gaussian_pyramid(image, levels)
    laplacian_pyramid = []

    for i in range(levels - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        expanded = cv.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        layer = cv.subtract(gaussian_pyramid[i], expanded)
        laplacian_pyramid.append(layer)

    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid


def guided_filter(guide, src, radius, eps):
    """
    guide: Изображение-проводник (оригинальный кадр или его уровень пирамиды)
    src: Фильтруемое изображение (карта резкости)
    radius: Радиус фильтра
    eps: Регуляризация (сглаживание)
    """
    guide = guide.astype(np.float32) / 255.0 if guide.max() > 1.0 else guide.astype(np.float32)
    src = src.astype(np.float32)

    mean_I = cv.boxFilter(guide, cv.CV_32F, (radius, radius))
    mean_p = cv.boxFilter(src, cv.CV_32F, (radius, radius))
    mean_Ip = cv.boxFilter(guide * src, cv.CV_32F, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv.boxFilter(guide * guide, cv.CV_32F, (radius, radius))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv.boxFilter(a, cv.CV_32F, (radius, radius))
    mean_b = cv.boxFilter(b, cv.CV_32F, (radius, radius))

    return mean_a * guide + mean_b

def reconstruct_from_pyramid(pyramid):
    """Собирает изображение обратно из пирамиды Лапласа."""
    levels = len(pyramid)
    current_image = pyramid[-1]
    
    for i in range(levels - 2, -1, -1):
        size = (pyramid[i].shape[1], pyramid[i].shape[0])
        current_image = cv.pyrUp(current_image, dstsize=size)
        current_image = cv.add(current_image, pyramid[i])
        
    return np.clip(current_image, 0, 255).astype(np.uint8)

def weighted_blending_at_level(layers, sharpness_maps):
    """
    Смешивает слои на основе их относительной резкости (мягкий выбор).
    Убирает резкие границы между кадрами.
    """
    max_map = np.max(sharpness_maps, axis=0)
    weights = np.exp(sharpness_maps - max_map) 
    weights /= (np.sum(weights, axis=0) + 1e-8)
    
    result = np.zeros_like(layers[0], dtype=np.float32)
    for i in range(len(layers)):
        w = np.expand_dims(weights[i], axis=-1) if len(layers[0].shape) == 3 else weights[i]
        result += layers[i] * w
        
    return result

def combine_images_pyramidal(images, levels=5, radius=7, eps=1e-2):
    num_images = len(images)
    all_pyramids = [build_laplacian_pyramid(img, levels) for img in images]
    
    final_pyramid = []
    
    for level_idx in range(levels):
        layers_at_level = [p[level_idx] for p in all_pyramids]
        
        sharpness_maps = [compute_sharpness_for_pyramid(img) for img in layers_at_level]
        
        refined_maps = []
        for i in range(num_images):
            guide = layers_at_level[i]
            if len(guide.shape) == 3:
                guide = cv.cvtColor(guide, cv.COLOR_BGR2GRAY)
            
            guide_norm = cv.normalize(guide, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)
            refined = guided_filter(guide_norm, sharpness_maps[i], radius, eps)
            refined_maps.append(refined)
            
        refined_maps = np.stack(refined_maps, axis=0)
        
        combined_layer = weighted_blending_at_level(layers_at_level, refined_maps)
        final_pyramid.append(combined_layer)
        
    return reconstruct_from_pyramid(final_pyramid)