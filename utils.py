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

    img = img.astype(np.float) / 255.0 # convert to float, then normalize
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert to grayscale

    # Define shifts for the FAST algorithm
    high_speed_first = [(0, 3), (0, -3)]       # Shifts for pixels 1 and 9
    high_speed_next = [(-3, 0), (3, 0)]        # Shifts for pixels 5 and 13
    remaining_points = [
        (-3, 1), (-2, 2), (-1, 3),
        (-1, -3), (-2, -2), (-3, -1),
        (1, 3), (2, 2), (3, 1),
        (1, -3), (3, -1), (2, -2)
    ]

    # Initialize bright and dark matrices
    brights = np.zeros_like(img_gray, dtype=np.float32)
    darks = np.zeros_like(img_gray, dtype=np.float32)

    # Apply high_speed_first shifts
    for shift in high_speed_first:
        shifted_img = np.roll(img_gray, shift=shift, axis=(0,1))
        brights += (shifted_img > (img_gray + THRESHOLD)).astype(np.float32)
        darks += (shifted_img < (img_gray - THRESHOLD)).astype(np.float32)

    # Determine possible corners
    possible_corners = (brights + darks) >= 1

    # Apply high_speed_next shifts on possible corners
    for shift in high_speed_next:
        shifted_img = np.roll(img_gray, shift=shift, axis=(0,1))
        brights += ((shifted_img > (img_gray + THRESHOLD)) & possible_corners).astype(np.float32)
        darks += ((shifted_img < (img_gray - THRESHOLD)) & possible_corners).astype(np.float32)

    # Apply remaining_points shifts on possible corners
    for shift in remaining_points:
        shifted_img = np.roll(img_gray, shift=shift, axis=(0,1))
        brights += ((shifted_img > (img_gray + THRESHOLD)) & possible_corners).astype(np.float32)
        darks += ((shifted_img < (img_gray - THRESHOLD)) & possible_corners).astype(np.float32)

    # Determine final corners
    corners = (brights >= N) | (darks >= N)

    # Create overlay by marking corners
    overlay = (img_gray * 255).astype(np.uint8)  # Convert back to 8-bit image
    overlay[corners] = 255  # Mark corner pixels as white

    return overlay