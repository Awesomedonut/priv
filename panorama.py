import cv2 as cv
import numpy as np

def find_homography_ransac(p1, p2, matches, ransac_params):
    """
    Find homography matrix using RANSAC
    """
    if len(matches) < 4:
        return None, []
    
    # Convert points to the format needed for findHomography
    if isinstance(matches, np.ndarray):
        # For FASTR case
        # Convert KeyPoints to coordinate arrays
        pts1 = np.float32([kp.pt for kp in p1])
        pts2 = np.float32([kp.pt for kp in p2])
        
        # Get the matching indices from the matches array
        if len(matches.shape) > 2:
            # If matches is a 3D array, find non-zero positions
            match_positions = np.nonzero(matches)
            pts1 = pts1[match_positions[0]]
            pts2 = pts2[match_positions[1]]
        else:
            # If matches is a 2D array of indices
            pts1 = pts1[matches[:, 0]]
            pts2 = pts2[matches[:, 1]]
    else:
        # For FAST case where matches are DMatch objects
        pts1 = np.float32([p1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([p2[m.trainIdx].pt for m in matches])
    
    # Reshape points for findHomography
    pts1 = pts1.reshape(-1, 1, 2)
    pts2 = pts2.reshape(-1, 1, 2)
    
    # Find homography matrix
    H, mask = cv.findHomography(
        pts1, 
        pts2,
        cv.RANSAC,
        ransacReprojThreshold=ransac_params['reprojection_threshold'],
        maxIters=ransac_params['max_iterations'],
        confidence=ransac_params['confidence']
    )
    
    return H, mask

def stitch_images(img1, img2, H):
    """
    Stitch two images using the computed homography matrix
    """
    if H is None:
        return None
    
    # Get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Find corners of first image after transformation
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = cv.perspectiveTransform(corners1, H)
    
    # Find the minimum and maximum coordinates
    all_corners = np.concatenate((
        corners2,
        np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    ))
    
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Translation matrix to move to positive coordinates
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    
    # Final dimensions
    width = xmax - xmin
    height = ymax - ymin
    
    # Create the panorama canvas
    panorama = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Warp first image
    warped1 = cv.warpPerspective(img1, Ht.dot(H), (width, height))
    
    # Warp second image
    warped2 = cv.warpPerspective(img2, Ht, (width, height))
    
    # Combine the images
    # Create masks for both images
    mask1 = warped1 > 0
    mask2 = warped2 > 0
    
    # Areas where both images overlap
    overlap = mask1 & mask2
    
    # Copy non-overlapping parts directly
    panorama[~overlap & mask1] = warped1[~overlap & mask1]
    panorama[~overlap & mask2] = warped2[~overlap & mask2]
    
    # Blend the overlapping regions
    panorama[overlap] = cv.addWeighted(
        warped1[overlap], 
        0.5,
        warped2[overlap], 
        0.5, 
        0
    )
    
    return panorama

def create_panorama(img1, img2, p1, p2, matches, ransac_params, is_fastr=False):
    """
    Create a panorama from two images using feature matches and RANSAC
    """
    if matches is None or len(matches) < 4:
        print("Not enough matches to create panorama")
        return None
    
    # Find homography
    H, mask = find_homography_ransac(p1, p2, matches, ransac_params)
    
    if H is None:
        print("Could not find valid homography")
        return None
    
    # Create panorama
    panorama = stitch_images(img1, img2, H)
    
    return panorama
