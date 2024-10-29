import cv2 as cv
import numpy as np
import random
from typing import Tuple, List, Optional
from fast_corner_detection import my_fast_detector, my_fastr_detector
from feature_matching import generate_descriptors_single_image, find_matches
import traceback

# RANSAC parameters for FAST
FAST_MAX_ITERATIONS = 2000
FAST_DISTANCE_THRESHOLD = 3.0

# RANSAC parameters for FASTR
FASTR_MAX_ITERATIONS = 2000
FASTR_DISTANCE_THRESHOLD = 3.0

def is_valid_homography(H: np.ndarray) -> bool:
    """Check if homography is valid."""
    # Check if matrix is finite
    if not np.isfinite(H).all():
        return False
        
    # Check determinant (should not be too close to zero)
    if abs(np.linalg.det(H)) < 1e-10:
        return False
        
    # Check scale (allow for more variation)
    scale = np.sqrt(np.sum(H[:2, :2]**2))
    if scale < 0.01 or scale > 100:
        return False
        
    return True

def create_panorama(im1: np.ndarray, im2: np.ndarray, use_fastr: bool = False) -> np.ndarray:
    if use_fastr:
        corners1 = my_fastr_detector(im1) == 255
        corners2 = my_fastr_detector(im2) == 255
        max_iter = FASTR_MAX_ITERATIONS
        dist_thresh = FASTR_DISTANCE_THRESHOLD
    else:
        corners1 = my_fast_detector(im1) == 255
        corners2 = my_fast_detector(im2) == 255
        max_iter = FAST_MAX_ITERATIONS
        dist_thresh = FAST_DISTANCE_THRESHOLD

    keypoints1, descriptors1 = generate_descriptors_single_image(im1, corners1, use_fastr)
    keypoints2, descriptors2 = generate_descriptors_single_image(im2, corners2, use_fastr)
    
    if descriptors1 is None or descriptors2 is None:
        print("No descriptors found!")
        return im1

    matches = find_matches(descriptors1, descriptors2)
    print(f"Number of matches found: {len(matches)}")
    
    if len(matches) < 4:
        print(f"Not enough matches found: {len(matches)}")
        return im1

    matched_points = extract_matched_points(keypoints1, keypoints2, matches)
    print(f"Number of matched points: {len(matched_points)}")
    
    if len(matched_points) < 4:
        print(f"Not enough matched points: {len(matched_points)}")
        return im1

    homography_matrix = find_best_homography(matched_points, max_iter, dist_thresh)
    if homography_matrix is None or not is_valid_homography(homography_matrix):
        print("Could not find valid homography")
        return im1

    print("Homography matrix:")
    print(homography_matrix)
    
    # Try original order
    result = stitch_ims(im1, im2, homography_matrix)
    if result is not None and not np.array_equal(result, im1):
        return result
    
    print("First attempt failed, trying reverse order...")
    
    # Try reverse order with inverse homography
    try:
        inv_homography = np.linalg.inv(homography_matrix)
        if is_valid_homography(inv_homography):
            result = stitch_ims(im2, im1, inv_homography)
            if result is not None and not np.array_equal(result, im2):
                return result
    except:
        pass
        
    print("Both stitching attempts failed")
    return im1

def stitch_ims(im1: np.ndarray, im2: np.ndarray, 
                 homography: np.ndarray) -> np.ndarray:
    try:
        # Get dimensions
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]
        
        print(f"Image 1 dimensions: {w1}x{h1}")
        print(f"Image 2 dimensions: {w2}x{h2}")

        # Calculate the size of the panorama
        corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners_transformed = cv.perspectiveTransform(corners, homography)
        print("Transformed corners:")
        print(corners_transformed.reshape(-1, 2))
        
        # Add padding to ensure we don't lose any image content
        padding = 100  # Add 100 pixels padding
        corners_all = np.concatenate((
            corners_transformed,
            np.float32([[0-padding, 0-padding], 
                       [0-padding, h2+padding], 
                       [w2+padding, h2+padding], 
                       [w2+padding, 0-padding]]).reshape(-1, 1, 2)
        ))
        
        [x_min, y_min] = np.int32(corners_all.min(axis=0).ravel())
        [x_max, y_max] = np.int32(corners_all.max(axis=0).ravel())
        
        # Translation matrix
        translation = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ])
        
        # Combine translation with homography
        final_matrix = translation.dot(homography)
        
        # Calculate size of output image
        panorama_width = x_max - x_min
        panorama_height = y_max - y_min
        
        print(f"Panorama dimensions: {panorama_width}x{panorama_height}")
        
        # Create the panorama
        panorama = cv.warpPerspective(im1, final_matrix, (panorama_width, panorama_height))
        
        # Copy the second image into the panorama
        y_offset = -y_min
        x_offset = -x_min
        
        # Create a mask for the second image
        mask = np.zeros((panorama_height, panorama_width), dtype=np.uint8)
        mask[y_offset:y_offset + h2, x_offset:x_offset + w2] = 1
        
        # Blend the images where they overlap
        panorama[y_offset:y_offset + h2, x_offset:x_offset + w2] = \
            cv.addWeighted(panorama[y_offset:y_offset + h2, x_offset:x_offset + w2], 
                          0.5,
                          im2,
                          0.5,
                          0)
        
        # Save intermediate results
        cv.imwrite('warped.png', panorama)
        
        # Crop black borders
        gray = cv.cvtColor(panorama, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if contours:
            x, y, w, h = cv.boundingRect(contours[0])
            panorama = panorama[y:y+h, x:x+w]
        
        return panorama

    except Exception as e:
        print(f"Error in stitch_ims: {str(e)}")
        traceback.print_exc()
        return None

def extract_matched_points(keypoints1: List, keypoints2: List, matches: List) -> List[List[float]]:
    matched_pts = []
    for match in matches:
        p1 = keypoints1[match.queryIdx].pt
        p2 = keypoints2[match.trainIdx].pt
        matched_pts.append([p1[0], p1[1], p2[0], p2[1]])
    return matched_pts

def compute_homography(points: List[List[float]]) -> Optional[np.ndarray]:
    try:
        if len(points) < 4:
            return None
            
        # Convert points to numpy arrays
        src_pts = np.float32([[point[0], point[1]] for point in points])
        dst_pts = np.float32([[point[2], point[3]] for point in points])
        
        # Use OpenCV's findHomography instead of manual computation
        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        
        if H is None or not np.isfinite(H).all():
            return None
            
        return H
        
    except Exception as e:
        print(f"Error computing homography: {str(e)}")
        return None

def find_best_homography(matched_points: List[List[float]], 
                        max_iterations: int = 5000,
                        distance_threshold: float = 5.0) -> Optional[np.ndarray]:
    if len(matched_points) < 4:
        return None

    # Convert points to numpy arrays
    src_pts = np.float32([[point[0], point[1]] for point in matched_points])
    dst_pts = np.float32([[point[2], point[3]] for point in matched_points])
    
    # Use OpenCV's findHomography with RANSAC
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 
                               ransacReprojThreshold=distance_threshold,
                               maxIters=max_iterations,
                               confidence=0.995)
    
    if H is None:
        return None
        
    # Count inliers
    num_inliers = np.sum(mask)
    print(f"Number of inliers: {num_inliers}")
    
    # Require at least 5% of matches to be inliers
    min_inliers = max(10, len(matched_points) * 0.05)
    if num_inliers < min_inliers:
        print(f"Too few inliers. Required: {min_inliers}")
        return None
        
    return H