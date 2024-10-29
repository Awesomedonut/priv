import cv2 as cv
from panorama import create_panorama
import time

def create_and_save_panoramas(img1_path: str, img2_path: str):
    # Load images
    left_img = cv.imread(img1_path)
    right_img = cv.imread(img2_path)
    
    # Create panorama with FAST parameters
    start_time = time.time()
    fast_result = create_panorama(left_img, right_img, fast_mode=True)
    fast_time = time.time() - start_time
    cv.imwrite('panorama_fast.jpg', fast_result)
    print(f"FAST panorama creation time: {fast_time:.2f} seconds")
    
    # Create panorama with FASTR parameters
    start_time = time.time()
    fastr_result = create_panorama(left_img, right_img, fast_mode=False)
    fastr_time = time.time() - start_time
    cv.imwrite('panorama_fastr.jpg', fastr_result)
    print(f"FASTR panorama creation time: {fastr_time:.2f} seconds")

if __name__ == "__main__":
    create_and_save_panoramas('IMG_1179.jpg', 'IMG_1182.jpg')
