import cv2 as cv
import numpy as np

from utils import read_and_resize_image, my_fast_detector


def main():
    # 1: Take 3 sets of 2 photographs to be stitched together to create a panorama
    photo_sets = {
        "IMG_1180.jpg": "S2-im1.png",
        "IMG_1181.jpg": "S2-im2.png",
        "IMG_1182.jpg": "S3-im1.png",
        "IMG_1179.jpg": "S3-im2.png"
    }

    for photo_path, img_name in photo_sets.items():
        read_and_resize_image(photo_path, img_name)
    
    # Step 2: Apply custom FAST detector
    # Note: Ensure that "S1-im1.png" exists in the working directory
    # fast_images = {
    #     "S1-fast.png": "S1-im1.png",
    #     "S2-fast.png": "S2-im1.png"
    # }
    
    # for output, input_path in fast_images.items():
    #     img = cv.imread(input_path)
    #     if img is None:
    #         print(f"Error: Unable to read image {input_path}")
    #         continue
    #     overlay = my_fast_detector(img)  # Using the refactored function
    #     cv.imwrite(output, overlay)
    #     print(f"Saved FAST overlay image to {output}")
    # Step 3: Stitch images together to create a panorama
    # panorama_images = {
    #     "panorama.png": "S1-im1.png", "S2-im1.png", "S3-im1.png"
    # }
    
    # for output, input_paths in panorama_images.items():
    #     img = cv.imread(input_paths[0])
    #     if img is None:
    #         print(f"Error: Unable to read image {input_paths[0]}")
    #         continue
    #     for input_path in input_paths[1:]:
    #         img = cv.imread(input_path)
    #         if img is None:
    #             print(f"Error: Unable to read image {input_path}")
    #             continue
    #         img = cv.stitch_images(img, img)
    #     cv.imwrite(output, img)
    #     print(f"Saved panorama image to {output}")

main()
