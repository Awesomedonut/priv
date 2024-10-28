import cv2 as cv
import numpy as np
import random

def create_panorama(img1, img2, keypoints1, keypoints2, matches, is_fastr=False):
    """
    Creates a panorama from two images using RANSAC for homography estimation
    """
    # RANSAC parameters
    if is_fastr:
        num_trials = 3000  # FASTR needs fewer trials due to better quality matches
        threshold = 4.0    # Stricter threshold for FASTR
    else:
        num_trials = 5000  # FAST needs more trials due to more noisy matches
        threshold = 5.0    # More lenient threshold for FAST

    # Extract matched points
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography using RANSAC
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 
                               ransacReprojThreshold=threshold,
                               maxIters=num_trials)

    if H is None:
        return None

    # Get dimensions of both images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Create points for corners of img1
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # Transform corners of img1
    transformed_corners = cv.perspectiveTransform(corners1, H)
    corners = np.concatenate((transformed_corners, corners2), axis=0)

    # Find min and max points to determine panorama size
    [x_min, y_min] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(corners.max(axis=0).ravel() + 0.5)

    # Create translation matrix to shift to positive coordinates
    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])

    # Combine translation with homography
    H_final = translation @ H

    # Create panorama
    panorama = cv.warpPerspective(img1, H_final, (x_max - x_min, y_max - y_min))
    
    # Copy img2 into the panorama
    panorama[-y_min:h2-y_min, -x_min:w2-x_min] = img2

    return panorama
