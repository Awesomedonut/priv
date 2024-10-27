import cv2 as cv
import numpy as np

THRESHOLD = 0.08
N = 12

def read_and_resize_image(img_path, img_name, size = (480, 750)): # size = (width, height)
    img = cv.imread(img_path)
    resized_img = cv.resize(img, size, interpolation=cv.INTER_AREA)
    cv.imwrite(img_name, resized_img)
    print(f"imwrite to {img_name}")

def my_fast_detector(img):
    normalized_img = normalize_image(img)
    img_gray = cv.cvtColor(normalized_img, cv.COLOR_BGR2GRAY)
    
    # Initialize matrices for counting bright and dark pixels
    brights, darks = initialize_detection_matrices(img_gray)
    
    # Get corner candidates using high-speed test points
    possible_corners = detect_initial_corners(img_gray, brights, darks)
    
    # Refine corner detection using remaining test points
    corners = detect_refined_corners(img_gray, brights, darks, possible_corners)
    
    return create_corner_overlay(img_gray, corners)

def normalize_image(img):
    return img.astype(np.float) / 255.0

def initialize_detection_matrices(img_gray):
    brights = np.zeros_like(img_gray, dtype=np.float32)
    darks = np.zeros_like(img_gray, dtype=np.float32)
    return brights, darks

def detect_initial_corners(img_gray, brights, darks):
    # High-speed test points (pixels 1, 9, 5, 13)
    high_speed_points = [(0, 3), (0, -3), (-3, 0), (3, 0)]
    
    for dx, dy in high_speed_points:
        shifted_img = np.roll(np.roll(img_gray, dx, axis=0), dy, axis=1)
        brights += (shifted_img > (img_gray + THRESHOLD)).astype(np.float32)
        darks += (shifted_img < (img_gray - THRESHOLD)).astype(np.float32)
    
    return (brights + darks) >= 1

def detect_refined_corners(img_gray, brights, darks, possible_corners):
    # Remaining test points in the Bresenham circle
    remaining_points = [
        (-3, 1), (-2, 2), (-1, 3),
        (-1, -3), (-2, -2), (-3, -1),
        (1, 3), (2, 2), (3, 1),
        (1, -3), (3, -1), (2, -2)
    ]
    
    for dx, dy in remaining_points:
        shifted_img = np.roll(np.roll(img_gray, dx, axis=0), dy, axis=1)
        mask = possible_corners & (shifted_img > (img_gray + THRESHOLD))
        brights += mask.astype(np.float32)
        mask = possible_corners & (shifted_img < (img_gray - THRESHOLD))
        darks += mask.astype(np.float32)
    
    return (brights >= N) | (darks >= N)

def create_corner_overlay(img_gray, corners):
    overlay = (img_gray * 255).astype(np.uint8)
    overlay[corners] = 255
    return overlay
