import cv2 as cv
import numpy as np
import time

from image_processing import read_and_resize_image
from fast_corner_detection import my_fast_detector, my_fastr_detector


def main():
    # 1: Take 3 sets of 2 photographs to be stitched together to create a panorama
    photo_sets = {
        "IMG_1180.jpg": "S2-im1.png",
        "IMG_1181.jpg": "S2-im2.png",
        "IMG_1182.jpg": "S3-im1.png",
        "IMG_1179.jpg": "S3-im2.png",
        "IMG_1185.jpg": "S4-im1.png",
        "IMG_1186.jpg": "S4-im2.png"
    }

    for photo_path, img_name in photo_sets.items():
        read_and_resize_image(photo_path, img_name)
    
    # 2: FAST feature detector
    imgs_for_fast_detection = {
        "S1-im1.png": "S1-fast.png",
        "S2-im1.png": "S2-fast.png"
    }
    
    for input_path, output in imgs_for_fast_detection.items():
        img = cv.imread(input_path)
        cv.imwrite(output, my_fast_detector(img))
        print(f"imwrite to {output}")

    # 2: FASTR feature detector with timing
    imgs_for_detection = {
        "S1-im1.png": ("S1-fast.png", "S1-fastR.png"),
        "S2-im1.png": ("S2-fast.png", "S2-fastR.png")
    }
    
    fast_times = []
    fastr_times = []
    
    for input_path, (fast_output, fastr_output) in imgs_for_detection.items():
        img = cv.imread(input_path)
        
        # Time FAST detection
        start_time = time.time()
        fast_result = my_fast_detector(img)
        fast_time = time.time() - start_time
        fast_times.append(fast_time)
        cv.imwrite(fast_output, fast_result)
        
        # Time FASTR detection
        start_time = time.time()
        fastr_result = my_fastr_detector(img)
        fastr_time = time.time() - start_time
        fastr_times.append(fastr_time)
        cv.imwrite(fastr_output, fastr_result)
        
        print(f"Processing {input_path}:")
        print(f"FAST time: {fast_time:.4f} seconds")
        print(f"FASTR time: {fastr_time:.4f} seconds")
    
    # Print average times
    avg_fast = sum(fast_times) / len(fast_times)
    avg_fastr = sum(fastr_times) / len(fastr_times)
    print(f"\nAverage FAST time: {avg_fast:.4f} seconds")
    print(f"Average FASTR time: {avg_fastr:.4f} seconds")

    # 3: Point description and matching
    # 4: RANSAC and Panoramas


main()
