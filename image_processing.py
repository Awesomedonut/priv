import cv2 as cv
import numpy as np

def read_and_resize_image(img_path, img_name, size=(480, 750)):  # size = (width, height)
    img = cv.imread(img_path)
    resized_img = cv.resize(img, size, interpolation=cv.INTER_AREA)
    cv.imwrite(img_name, resized_img)
    print(f"imwrite to {img_name}")

def convert_img_to_normalized_float(img):  # this is the same as im2double since float is 64-bit in python
    return img.astype(float) / 255.0
