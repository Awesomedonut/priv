import cv2 as cv
import numpy as np

def describe_and_match_features(img1, img2, corners1, corners2):
    """
    Detect, describe and match features between two images using ORB
    """
    # Convert images to grayscale
    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    
    # Initialize ORB detector
    orb = cv.ORB_create(nfeatures=10000)
    
    # Print number of corners detected
    print(f"Number of corners in image 1: {np.sum(corners1)}")
    print(f"Number of corners in image 2: {np.sum(corners2)}")
    
    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(img1_gray, mask=corners1.astype(np.uint8))
    kp2, des2 = orb.detectAndCompute(img2_gray, mask=corners2.astype(np.uint8))
    
    # Print number of keypoints detected
    print(f"Number of keypoints in image 1: {len(kp1) if kp1 else 0}")
    print(f"Number of keypoints in image 2: {len(kp2) if kp2 else 0}")
    
    # Check if we found any keypoints and descriptors
    if not kp1 or not kp2 or des1 is None or des2 is None:
        print("No keypoints or descriptors found")
        return None
    
    try:
        # Create BFMatcher object
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        
        # Match descriptors
        matches = matcher.match(des1, des2)
        
        if len(matches) == 0:
            print("No matches found")
            return None
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Draw top 50 matches instead of 20
        num_matches_to_draw = min(50, len(matches))
        matched_img = cv.drawMatches(
            img1, kp1,
            img2, kp2,
            matches[:num_matches_to_draw], None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # Resize if the image is too large
        if matched_img.shape[1] > 1200:
            height = int(1200 * matched_img.shape[0] / matched_img.shape[1])
            matched_img = cv.resize(matched_img, (1200, height))
            
        print(f"Found {len(matches)} matches, showing top {num_matches_to_draw}")
        print(f"Best match distance: {matches[0].distance:.2f}")
        print(f"Worst shown match distance: {matches[num_matches_to_draw-1].distance:.2f}")
        
        return matched_img
        
    except Exception as e:
        print(f"Error during matching: {str(e)}")
        return None
