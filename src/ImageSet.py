import numpy as np
import os
import cv2 as cv

class ImagesSet:
    def __init__(self, folder_path, sample_image_path):
        self.images = self.load_images(folder_path)
        self.sample_image = self.load_image(sample_image_path)

    def load_images(self, folder_path):
        files = os.listdir(folder_path)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        try:
            image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        except ValueError:
            raise ValueError("Failed to sort images")

        images = []
        images_size = None
        for filename in image_files:
            image_path = os.path.join(folder_path, filename)
            img = cv.imread(image_path)
            if images_size != None:
                if img.shape != images_size :
                    raise ValueError("Images of different sizes are provided, but images of the same size are required")
            else:
                images_size = img.shape

            if img is not None:
                images.append(img)
            else:
                print(f"Не удалось загрузить изображение: {image_path}")
        return np.array(images)

    def load_image(self, image_path):
        img = cv.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Не удалось загрузить образец: {image_path}")
        return img