import cv2 as cv
import numpy as np
import time

from image_processing import read_and_resize_image
from fast_corner_detection import my_fast_detector, my_fastr_detector
from feature_matching import generate_descriptors, generate_descriptors_single_image, match_features
from timing import calculate_and_print_average_times, print_matching_times
from panorama import compute_matched_points, create_panorama


def main():
    # 1: Take 3 sets of 2 photographs to be stitched together to create a panorama
    photo_sets = {
        # "IMG_1180.jpg": "S2-im1.png",
        # "IMG_1181.jpg": "S2-im2.png", 
        # "IMG_1182.jpg": "S3-im1.png",
        # "IMG_1179.jpg": "S3-im2.png",
        "IMG_1203.jpg": "S2-im1.png",
        "IMG_1204.jpg": "S2-im2.png",
        "IMG_1207.jpg": "S3-im1.png", 
        "IMG_1208.jpg": "S3-im2.png",
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

    # 2: Robust FAST using Harris Cornerness metric (FASTR)
    imgs_for_detection = {
        "S1-im1.png": ("S1-fast.png", "S1-fastR.png"),
        "S2-im1.png": ("S2-fast.png", "S2-fastR.png")
    }
    
    fast_times = []
    fastr_times = []
    
    for input_path, (fast_output, fastr_output) in imgs_for_detection.items():
        img = cv.imread(input_path)
        
        # timing FAST detection
        start_time = time.time()
        fast_result = my_fast_detector(img)
        fast_time = time.time() - start_time
        fast_times.append(fast_time)
        cv.imwrite(fast_output, fast_result)
        
        # timing FASTR detection
        start_time = time.time()
        fastr_result = my_fastr_detector(img)
        fastr_time = time.time() - start_time
        fastr_times.append(fastr_time)
        cv.imwrite(fastr_output, fastr_result)
        
        print(f"FAST time: {fast_time:.4f} seconds")
        print(f"FASTR time: {fastr_time:.4f} seconds")
    
    calculate_and_print_average_times(fast_times, fastr_times)

    # 3: Point description and matching
    matching_sets = {
        ("S3-im1.png", "S3-im2.png"): ("S1-fastMatch.png", "S1-fastRMatch.png"),
        ("S2-im1.png", "S2-im2.png"): ("S2-fastMatch.png", "S2-fastRMatch.png")
    }
    
    fast_desc_times = []
    fast_match_times = []
    fastr_desc_times = []
    fastr_match_times = []
    
    for (img1_path, img2_path), (fast_match_output, fastr_match_output) in matching_sets.items():
        img1 = cv.imread(img1_path)
        img2 = cv.imread(img2_path)
        
        fast_corners1 = my_fast_detector(img1) == 255
        fast_corners2 = my_fast_detector(img2) == 255
        
        fastr_corners1 = my_fastr_detector(img1) == 255
        fastr_corners2 = my_fastr_detector(img2) == 255
        
        print(f"\nprocessing {img1_path} and {img2_path}")
        
        # FAST corners
        p1, p2, des1, des2, desc_time = generate_descriptors(img1, img2, fast_corners1, fast_corners2, is_fastr=False)
        fast_desc_times.append(desc_time)
        
        if all(x is not None for x in [p1, p2, des1, des2]):
            fast_matches, match_time = match_features(img1, img2, p1, p2, des1, des2, is_fastr=False)
            fast_match_times.append(match_time)
            if fast_matches is not None:
                cv.imwrite(fast_match_output, fast_matches)
        
        # FASTR corners
        p1, p2, des1, des2, desc_time = generate_descriptors(img1, img2, fastr_corners1, fastr_corners2, is_fastr=True)
        fastr_desc_times.append(desc_time)
        
        if all(x is not None for x in [p1, p2, des1, des2]):
            fastr_matches, match_time = match_features(img1, img2, p1, p2, des1, des2, is_fastr=True)
            fastr_match_times.append(match_time)
            if fastr_matches is not None:
                cv.imwrite(fastr_match_output, fastr_matches)
    
    print_matching_times(fast_desc_times, fast_match_times, fastr_desc_times, fastr_match_times)

    # 4: RANSAC and Panoramas
    panorama_sets = [
        ("S3-im1.png", "S3-im2.png", "S3-panorama.png"),
        ("S2-im1.png", "S2-im2.png", "S2-panorama.png"),
        ("S4-im1.png", "S4-im2.png", "S4-panorama.png")
    ]
    
    for img1_path, img2_path, panorama_output in panorama_sets:
        img1 = cv.imread(img1_path)
        img2 = cv.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"Could not read images {img1_path} or {img2_path}")
            continue
            
        # Create panorama using SIFT features
        panorama = create_panorama(img1, img2, None, None)  # p1, p2 are not used anymore
        
        if panorama is not None:
            cv.imwrite(panorama_output, panorama)
            print(f"Panorama saved to {panorama_output}")
        else:
            print(f"Failed to create panorama for {img1_path} and {img2_path}")
   


  

    #5: Stitch 4 images instead of 2

if __name__ == "__main__":
    main()
