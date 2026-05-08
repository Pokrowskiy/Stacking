import cv2 as cv
import numpy as np

def align_images(images, reference_idx=0):
    """
    Выравнивает список изображений относительно эталонного кадра.
    Использует алгоритм ECC (Enhanced Correlation Coefficient).

    images: np.array с изображеними для выравнивания
    reference_idx: индекс эталонного кадра
    Возвращает np.array с выровненными изображениями.
    """
    aligned_images = []

    ref_img = images[reference_idx]
    ref_gray = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)

    warp_mode = cv.MOTION_HOMOGRAPHY

    if warp_mode == cv.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    number_of_iterations = 300
    termination_eps = 1e-6
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    print(f"Starting alignment relative to image index {reference_idx}...")

    for i, img in enumerate(images):
        if i == reference_idx:
            aligned_images.append(img)
            continue

        curr_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        try:
            (cc, warp_matrix) = cv.findTransformECC(ref_gray, curr_gray, warp_matrix, warp_mode, criteria)

            if warp_mode == cv.MOTION_HOMOGRAPHY:
                img_aligned = cv.warpPerspective(img, warp_matrix, (ref_img.shape[1], ref_img.shape[0]),
                                                flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
            else:
                img_aligned = cv.warpAffine(img, warp_matrix, (ref_img.shape[1], ref_img.shape[0]),
                                           flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

            aligned_images.append(img_aligned)
            print(f"Image {i} aligned successfully.")

        except cv.error as e:
            print(f"Warning: Image {i} could not be aligned. Using original. Error: {e}")
            aligned_images.append(img)

    return np.array(aligned_images)

def crop_artifacts(images, p=0.05):
    """Обрезка краев на p процентов для удаления мусора от ECC."""
    h, w = images[0].shape[:2]
    dh, dw = int(h * p), int(w * p)
    return [img[dh:h-dh, dw:w-dw] for img in images]

def denoise_image_cv(image):
    """Функция денойзинга с заранее подобранными параметрами"""
    denoised = cv.fastNlMeansDenoisingColored(
        image,
        h=5,
        hColor=5,
        templateWindowSize=7,
        searchWindowSize=11
    )
    return denoised