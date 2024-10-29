import cv2 as cv
import numpy as np
import random
from typing import Tuple, List, Optional
from fast_corner_detection import my_fast_detector, my_fastr_detector
from feature_matching import generate_descriptors_single_image

class RANSACParams:
    def __init__(self, max_iterations: int, distance_threshold: float):
        self.max_iterations = max_iterations
        self.distance_threshold = distance_threshold

# Adjusted RANSAC parameters
FAST_PARAMS = RANSACParams(max_iterations=1000, distance_threshold=5.0)
FASTR_PARAMS = RANSACParams(max_iterations=1000, distance_threshold=10.0)

def create_panorama(im1: np.ndarray, im2: np.ndarray, use_fastr: bool = True) -> np.ndarray:
    """Create panorama with either FAST or FASTR parameters"""
    keypoints_data = extract_and_match_keypoints(im1, im2, use_fastr)
    params = FASTR_PARAMS if use_fastr else FAST_PARAMS
    homography_matrix = find_best_homography(
        keypoints_data, 
        max_iterations=params.max_iterations,
        distance_threshold=params.distance_threshold
    )
    return stitch_ims(im1, im2, homography_matrix)

def extract_and_match_keypoints(im1: np.ndarray, im2: np.ndarray, use_fastr: bool) -> List[List[float]]:
    # Use our custom FAST/FASTR detector
    corners1 = my_fastr_detector(im1) if use_fastr else my_fast_detector(im1)
    corners2 = my_fastr_detector(im2) if use_fastr else my_fast_detector(im2)
    
    # Get keypoints and descriptors using our feature matching functions
    keypoints1, descriptors1 = generate_descriptors_single_image(im1, corners1 == 255, use_fastr)
    keypoints2, descriptors2 = generate_descriptors_single_image(im2, corners2 == 255, use_fastr)
    
    return match_keypoints(keypoints1, keypoints2, descriptors1, descriptors2)

def match_keypoints(keypoints1: List, keypoints2: List, 
                   descriptors1: np.ndarray, descriptors2: np.ndarray) -> List[List[float]]:
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    matched_pts = []
    for match in matches:
        p1 = keypoints1[match.queryIdx].pt
        p2 = keypoints2[match.trainIdx].pt
        matched_pts.append([p1[0], p1[1], p2[0], p2[1]])

    return matched_pts

def compute_homography(points: List[List[float]]) -> np.ndarray:
    equation_matrix = []
    for point in points:
        x1, y1 = point[0], point[1]
        x2, y2 = point[2], point[3]
        equation_matrix.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        equation_matrix.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])

    equation_matrix = np.array(equation_matrix)
    _, _, vh = np.linalg.svd(equation_matrix)
    homography = vh[-1, :].reshape(3, 3)
    return homography / homography[2, 2]

def find_best_homography(matched_points: List[List[float]], 
                        max_iterations: int,
                        distance_threshold: float) -> np.ndarray:
    best_inliers = []
    best_homography = None
    
    for _ in range(max_iterations):
        sample_points = random.choices(matched_points, k=4)
        homography = compute_homography(sample_points)
        current_inliers = find_inliers(matched_points, homography, distance_threshold)
        
        if len(current_inliers) > len(best_inliers):
            best_inliers = current_inliers
            best_homography = homography
    
    return best_homography

def find_inliers(points: List[List[float]], 
                 homography: np.ndarray,
                 threshold: float) -> List[List[float]]:
    inliers = []
    for point in points:
        p1 = np.array([point[0], point[1], 1]).reshape(3, 1)
        p2 = np.array([point[2], point[3], 1]).reshape(3, 1)
        
        transformed_p1 = np.dot(homography, p1)
        transformed_p1 = transformed_p1 / transformed_p1[2]
        
        if np.linalg.norm(p2 - transformed_p1) < threshold:
            inliers.append(point)
    
    return inliers

def stitch_ims(im1: np.ndarray, im2: np.ndarray, homography: np.ndarray) -> np.ndarray:
    # Swap height/width assignments to match image order
    height1, width1 = im1.shape[:2]  # Changed from im2 to im1
    height2, width2 = im2.shape[:2]  # Changed from im1 to im2

    # Create corners for both images
    corners1 = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)
    
    # Transform corners
    transformed_corners2 = cv.perspectiveTransform(corners2, homography)
    all_corners = np.concatenate((corners1, transformed_corners2), axis=0)

    # Find the dimensions of the stitched image
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())

    # Calculate offset
    offset = [-x_min, -y_min]

    # Create the translation matrix
    translation_matrix = np.array([
        [1, 0, offset[0]],
        [0, 1, offset[1]],
        [0, 0, 1]
    ])

    # Warp and create the panorama
    final_homography = translation_matrix.dot(homography)
    output_size = (x_max - x_min, y_max - y_min)
    
    # Create the warped image
    warped_image = cv.warpPerspective(im2, final_homography, output_size)  # Changed from im1 to im2
    
    # Copy the first image into the warped image
    warped_image[offset[1]:offset[1] + height1, 
                offset[0]:offset[0] + width1] = im1  # Changed from im2 to im1
    
    return warped_image