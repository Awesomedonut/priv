import cv2
import numpy as np

def compute_homography(p1, p2):
    """
    Computes homography using RANSAC with adjusted parameters.
    
    :param p1: Array of points from image 1.
    :param p2: Array of points from image 2.
    :return: Homography matrix H and mask.
    """
    if len(p1) < 4 or len(p2) < 4:
        return None, None
        
    # Adjust RANSAC parameters
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 
                                ransacReprojThreshold=5.0,  # Increased threshold
                                maxIters=2000,  # More iterations
                                confidence=0.995)  # Higher confidence
    
    if H is None:
        return None, None
        
    # Count inliers
    num_inliers = np.sum(mask)
    print(f"Number of inliers: {num_inliers}")
    
    if num_inliers < 10:  # Minimum number of inliers needed
        return None, None
        
    return H, mask

def warp_images(img1, img2, H):
    """
    Warps img2 onto img1 using homography H.
    
    :param img1: First image.
    :param img2: Second image.
    :param H: Homography matrix.
    :return: Resulting panorama image.
    """
    # Get dimensions of the input images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Get the canvas dimensions
    # Warp the corners of img2 to find size of panorama
    corners_img2 = np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)
    warped_corners_img2 = cv2.perspectiveTransform(corners_img2, H)
    
    # Combine corners from img1 and warped img2 to get the output canvas size
    corners = np.concatenate((np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2), warped_corners_img2), axis=0)
    
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)
    
    # Translation matrix to shift the panorama
    translation = [-xmin, -ymin]
    H_translation = np.array([[1, 0, translation[0]],
                              [0, 1, translation[1]],
                              [0, 0, 1]])
    
    # Warp the second image
    result = cv2.warpPerspective(img2, H_translation.dot(H), (xmax - xmin, ymax - ymin))
    
    # Paste the first image into the panorama
    result[translation[1]:h1+translation[1], translation[0]:w1+translation[0]] = img1
    
    return result

def create_panorama(img1, img2, p1, p2):
    """
    Creates a panorama from two images using SIFT features and RANSAC
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    # Create FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Find matches
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
            
    print(f"Number of good matches: {len(good_matches)}")
    
    if len(good_matches) < 10:
        return None
        
    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        return None
    
    # Get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Get corners of img1
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2_transformed = cv2.perspectiveTransform(corners1, H)
    
    # Get corners of both images
    corners = np.concatenate((
        np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2),
        corners2_transformed
    ))
    
    # Get the canvas dimensions
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)
    
    # Translation matrix
    translation = np.array([
        [1, 0, -xmin],
        [0, 1, -ymin],
        [0, 0, 1]
    ])
    
    # Warp img1
    output_img = cv2.warpPerspective(
        img1, 
        translation.dot(H),
        (xmax - xmin, ymax - ymin)
    )
    
    # Copy img2 into the output image
    output_img[-ymin:h2-ymin, -xmin:w2-xmin] = img2
    
    return output_img

def compute_matched_points(p1, p2, des1, des2):
    """
    Computes matched points between two images using improved matching strategy.
    """
    # Create BFMatcher object for k-nearest neighbor matching
    bf = cv2.BFMatcher()
    
    # Find k=2 nearest matches for each descriptor
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test with more lenient threshold
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:  # More lenient ratio test
            good_matches.append(m)
    
    print(f"Number of matches found: {len(good_matches)}")
    
    if len(good_matches) < 10:
        return [], []
    
    # Extract matched points
    matched_pts1 = []
    matched_pts2 = []
    for match in good_matches:
        matched_pts1.append(p1[match.queryIdx].pt)
        matched_pts2.append(p2[match.trainIdx].pt)
    
    return np.float32(matched_pts1), np.float32(matched_pts2)
