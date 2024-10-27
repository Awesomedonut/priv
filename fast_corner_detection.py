import cv2 as cv
import numpy as np
from image_processing import convert_img_to_normalized_float

# values from FAST wikipedia page
THRESHOLD = 0.08
N = 12
# high-speed test points: example pixels 1, 9, 5, 13
high_speed_pixels = [(0, 3), (0, -3), (-3, 0), (3, 0)]

# harris constants
K = 0.04
H_THRESH = 0.01

def my_fast_detector(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = convert_img_to_normalized_float(img_gray)
    
    brights, darks = initialize_matrices(img_gray)
    corners = detect_initial_corners(img_gray, brights, darks)
    corners = detect_other_corners(img_gray, brights, darks, corners)
    corners = create_corner_overlay(img_gray, corners)
    
    return corners

def initialize_matrices(img_gray):
    brights = np.zeros_like(img_gray, dtype=float) # creates matrix of zeroes
    darks = np.zeros_like(img_gray, dtype=float)
    return brights, darks

def detect_initial_corners(img_gray, brights, darks):
    for dx, dy in high_speed_pixels:
        shifted_img = np.roll(np.roll(img_gray, dx, axis=0), dy, axis=1)
        brights += (shifted_img > (img_gray + THRESHOLD)).astype(float)
        darks += (shifted_img < (img_gray - THRESHOLD)).astype(float)
    
    return (brights + darks) >= 1

def detect_other_corners(img_gray, brights, darks, possible_corners):
    # remaining points that aren't high-speed test points
    remaining_points = [
        (-3, 1), (-2, 2), (-1, 3),
        (-1, -3), (-2, -2), (-3, -1),
        (1, 3), (2, 2), (3, 1),
        (1, -3), (3, -1), (2, -2)
    ]
    
    for dx, dy in remaining_points:
        shifted_img = np.roll(np.roll(img_gray, dx, axis=0), dy, axis=1)
        mask = possible_corners & (shifted_img > (img_gray + THRESHOLD))
        brights += mask.astype(float)
        mask = possible_corners & (shifted_img < (img_gray - THRESHOLD))
        darks += mask.astype(float)
    
    return (brights >= N) | (darks >= N)

def create_corner_overlay(img_gray, corners):
    overlay = (img_gray * 255).astype(np.uint8)  # convert to int so it's compatible with original image
    overlay[corners] = 255
    return overlay

def my_fastr_detector(img):
    """FAST detector with Harris corner measure filtering"""
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = convert_img_to_normalized_float(img_gray)
    
    # Get FAST corners first
    brights, darks = initialize_matrices(img_gray)
    corners = detect_initial_corners(img_gray, brights, darks)
    corners = detect_other_corners(img_gray, brights, darks, corners)
    
    # Apply Harris corner measure
    corner_mask = apply_harris_measure(img_gray, corners)
    
    # Create visualization
    corners = create_corner_overlay(img_gray, corner_mask)
    return corners

def apply_harris_measure(img_gray, fast_corners):
    """Calculate Harris corner measure for FAST points"""
    # Calculate derivatives
    dx = cv.Sobel(img_gray, cv.CV_64F, 1, 0, ksize=3)
    dy = cv.Sobel(img_gray, cv.CV_64F, 0, 1, ksize=3)
    
    # Compute components of the Harris matrix
    dx2 = dx * dx
    dy2 = dy * dy
    dxy = dx * dy
    
    # Gaussian smoothing
    window_size = 3
    dx2 = cv.GaussianBlur(dx2, (window_size, window_size), 0)
    dy2 = cv.GaussianBlur(dy2, (window_size, window_size), 0)
    dxy = cv.GaussianBlur(dxy, (window_size, window_size), 0)
    
    # Calculate Harris response
    det = (dx2 * dy2) - (dxy * dxy)
    trace = dx2 + dy2
    harris_response = det - K * (trace * trace)
    
    # Only consider Harris response at FAST corner points
    harris_response[~fast_corners] = 0
    
    # Threshold Harris response
    strong_corners = harris_response > H_THRESH
    
    return strong_corners
