import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def grayscale_an_image(original_image):
    # read image as grayscale
    # grayscale = cv2.imread("tokyo_tower.png", cv2.IMREAD_GRAYSCALE)

    # or convert an original image
    return cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

def scale_image(original_image: np.ndarray, horizontal_value = 0.5, vertical_value = 1.5):
    # Scale image (50% horizontally, 150% vertically) 
    M = np.float32([[horizontal_value, 0, 0], [0, vertical_value, 0]])

    original_height, original_width, c = original_image.shape
    
    width = original_width * horizontal_value
    height = original_height * vertical_value

    return cv2.warpAffine(original_image, M, (int(width), int(height)))

def scale_images(images: list, horizontal_value = 0.5, vertical_value = 0.5, suffix = "_half"):
    '''
    Resize collection of images (list of paths) and saves with new suffix
    '''
    M = np.float32([[horizontal_value, 0, 0], [0, vertical_value, 0]])

    for i in images:
        img = cv2.imread(i)
        original_height, original_width, c = img.shape
    
        width = original_width * horizontal_value
        height = original_height * vertical_value

        img_resized = cv2.warpAffine(img, M, (int(width), int(height)))

        title, ext = os.path.splitext(i)

        cv2.imwrite(f"{title}{suffix}{ext}", img_resized)

def resize_image(original_image: np.ndarray, width: int = 275, height: int = 375):
    return cv2.resize(original_image, (width, height))

def translate_image(original_image: np.ndarray, horizonatl_value = 100, vertical_value = 50):
    # Translate image (100 horizontally, 50 vertically) 
    M = np.float32([[1, 0, horizonatl_value], [0, 1, vertical_value]])

    height, width, c = original_image.shape

    return cv2.warpAffine(original_image, M, (width, height))

def rotate_image(original_image: np.ndarray, angle: int = 90):
    '''
    Find the matrix for rotating the image 90 degrees counterclockwise
    with the rotation center coordinates as the center of the image
    '''

    orig_h, orig_w, c = original_image.shape

    is_height_greater_than_width = orig_h > orig_w
    diff = orig_h - orig_w if is_height_greater_than_width else orig_w - orig_h
    diff /= 2
    new_h = orig_h if is_height_greater_than_width else orig_w
    new_w = orig_h if is_height_greater_than_width else orig_w

    # ①Move left by 125 and expand display area to (750, 750)=(h, h)
    M0 = np.float32([[1, 0, diff], [0, 1, 0]])
    moved_image = cv2.warpAffine(original_image, M0, (new_w, new_h))

    # ② Center point is (375, 375)=(w/2+125, h/2), rotate 90 degrees counterclockwise 
    M1 = cv2.getRotationMatrix2D((orig_w/2 + diff, orig_h/2), angle, 1)
    # Output image size is (750, 750)=(h, h)
    rotated_image = cv2.warpAffine(moved_image, M1, (new_w, new_h))

    # ③ Move up 125 (-125 vertically) to make the display area (750,500)=(h, w)
    M2 = np.float32([[1, 0, 0], [0, 1, -diff]])

    return cv2.warpAffine(rotated_image, M2, (orig_h, orig_w))

def flip_image(original_image: np.ndarray, direction: int = 0):
    return cv2.flip(original_image, direction)

def main():
    # reads as BGR on default
    # tokyo_tower = cv2.imread("tokyo_tower.png")

    # convert BGR to RGB
    # rgb_tower = cv2.cvtColor(tokyo_tower, cv2.COLOR_BGR2RGB)

    # save image with opencv
    # cv2.imwrite("tokyo_tower_gray.png", grayscale_an_image(tokyo_tower))

    tokyo_tower = cv2.cvtColor(cv2.imread("tokyo_tower.png"), cv2.COLOR_BGR2RGB)

    # get all files in dir with glob
    files = glob.glob('./images/*.JPG')

    scale_images(files)

    # plt.imshow(flip_image(tokyo_tower))

    # plt.show()

main()
