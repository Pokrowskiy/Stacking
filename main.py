import os
import time
import argparse
import cv2 as cv
import psutil
import tracemalloc
import torch

from src.ImageSet import ImagesSet
from src.Utils import align_images, crop_artifacts
from src.BasicStacking import combine_images_basic
from src.PyramidalStacking import combine_images_pyramidal
from src.Metrics import compute_difference_metric, compute_difference_image
from src.UNet_stacking import combine_images_ml

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    parser = argparse.ArgumentParser(description="Focus Stacking CLI Tool")
    parser.add_argument('--method', type=str, choices=['basic', 'pyramidal', 'ML'], default=None)
    parser.add_argument('--images_dir', type=str, default='samples/images')
    parser.add_argument('--output_dir', type=str, default='samples/output')
    parser.add_argument('--use_metrics', type=str, help='Path to expected.jpg')
    parser.add_argument('--console_timer_output', type=bool, default=True)

    args = parser.parse_args()

    mem_start = get_memory_usage()
    if args.console_timer_output:
        print(f"Начальное потребление памяти: {mem_start:.2f} MB")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    start_time = time.time()

    img_set = ImagesSet(args.images_dir, args.use_metrics)
    images_orig = img_set.images
    
    if args.console_timer_output:
        print(f"Загружено изображений: {len(images_orig)}. Время: {time.time()-start_time:.2f}s")
        print(f"Потребление памяти для загрузки изображений: {get_memory_usage()-mem_start:.2f} MB")

    t = time.time()
    tracemalloc.start()
    ref_idx = len(images_orig) // 2
    images_aligned_raw = align_images(images_orig, reference_idx=ref_idx)
    if args.console_timer_output:
        print(f"Выравнивание изображений завершено. Время: {time.time()-t:.2f}s")
    

    images_aligned = crop_artifacts(images_aligned_raw, p=0.05)
    t = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if args.console_timer_output:
        print(f"Обрезка изображений завершена. Время: {time.time()-t:.2f}s")
        print(f"Пиковое потребление памяти на выравнивании: {peak / 1024 / 1024:.2f} MB")

    methods = [args.method] if args.method else ['basic', 'pyramidal']

    methods = [args.method] if args.method else ['basic', 'pyramidal', 'ML']

    for method in methods:
        t_m = time.time()
        tracemalloc.start()
        
        # Сброс пика памяти GPU перед замером метода
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        result = None
        
        if method == 'basic':
            result = combine_images_basic(images_aligned, images_aligned)
        elif method == 'pyramidal':
            result = combine_images_pyramidal(images_aligned, levels=5)
        elif method == 'ML':
            model_weights = "models/model_weights.pth"
            result = combine_images_ml(images_aligned, model_path=model_weights)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        if result is not None:
            out_name = f"result_{method}.png"
            cv.imwrite(os.path.join(args.output_dir, out_name), result)
            
            if args.console_timer_output:
                print(f"Метод {method}: Закончил обработку за {time.time()-t_m:.2f}s")
                print(f"Метод {method}: Пик ОЗУ (CPU): {peak / 1024 / 1024:.2f} MB")
                
                if method == 'ML' and torch.cuda.is_available():
                    gpu_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
                    print(f"Метод {method}: Пик видеопамяти (GPU VRAM): {gpu_peak:.2f} MB")

            if args.use_metrics:
                sample_cropped = crop_artifacts([img_set.sample_image], p=0.05)[0]
                cv.imwrite(os.path.join(args.output_dir, "cropped_sample.png"), sample_cropped)
                cv.imwrite(os.path.join(args.output_dir, f"diff_image_{method}.png"), compute_difference_image(result, sample_cropped))
                m = compute_difference_metric(result, sample_cropped)
                print(f"Метрики {method}: PSNR: {m['PSNR']:.2f}, SSIM: {m['SSIM']:.4f}, LPIPS: {m['LPIPS']:.4f}")

if __name__ == "__main__":
    main()