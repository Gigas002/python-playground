import numpy as np
import cv2
import matplotlib.pyplot as plt


def demonstrate_hsv():
    h_map = np.zeros((60, 180, 3), dtype=np.uint8)

    # H (hue) is a continuous integer from 0 to 179
    for i in range(180):
        h_map[:, i, 0] = i

    # Set all S (saturation) to 255
    h_map[:, :, 1] = 255

    # Set all V (brightness) to 255
    h_map[:, :, 2] = 255

    h_map = cv2.cvtColor(h_map, cv2.COLOR_HSV2BGR)

    fig = plt.figure(figsize=(3, 1), dpi=200)

    plt.imshow(cv2.cvtColor(h_map, cv2.COLOR_BGR2RGB))
    plt.show()


def make_mask(base_img):
    # convert to HSV
    base_img_hsv = cv2.cvtColor(base_img, cv2.COLOR_BGR2HSV)
    h, w, c = base_img.shape

    # lower and upper limits of H, S, V
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])

    # binarize each pixel, according to whether it falls between lower and upper blue spectre
    mask = cv2.inRange(base_img_hsv, lower_blue, upper_blue)

    # mask array
    mask_img = np.zeros((h, w, c), dtype=np.uint8)

    for i in range(c):
        mask_img[:, :, i] = mask

    return mask_img


def binarize_mask(masked_img):
    masked_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    ret, masked_gray = cv2.threshold(masked_gray, 0, 255, cv2.THRESH_BINARY)

    return masked_gray


def get_contours(original_image, binarized_mask):
    contours, hierarchy = cv2.findContours(
        binarized_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contoured_img = np.copy(original_image)

    # sort out some garbage
    for i in range(len(contours)):
        # ignore data with small area as garbage data
        contour = contours[i]
        c_area = cv2.contourArea(contour)

        if (c_area < 1000):
            continue

        cv2.polylines(contoured_img, contours[i], True, (0, 0, 255), 2)

    return contoured_img


def rectangle_contour(original_image, binarized_mask):
    rect = np.copy(original_image)

    contours, hierarchy = cv2.findContours(
        binarized_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        # ignore garbage data
        contour = contours[i]
        c_area = cv2.contourArea(contour)

        if c_area < 1000:
            continue

        # get top-left coords and width/height information about ball
        x, y, w, h = cv2.boundingRect(contour)

        # enclose ball into rectangle
        cv2.rectangle(rect, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return rect


def main():
    # demonstrate_hsv()

    ball_img = cv2.imread("home/home2.png")

    masked_img = make_mask(ball_img)
    # masked_object = cv2.bitwise_and(ball_img, masked_img)
    binarized_mask = binarize_mask(masked_img)

    # plt.imshow(cv2.cvtColor(masked_object, cv2.COLOR_BGR2RGB))
    # plt.imshow(binarize_mask(masked_img), cmap="gray")

    plt.imshow(cv2.cvtColor(rectangle_contour(
        ball_img, binarized_mask), cv2.COLOR_BGR2RGB))

    plt.show()


def smoothing(noise_img):
    kernel = np.ones((3, 3)) / 9.0

    return cv2.filter2D(noise_img, -1, kernel)


def blur(noise_img):
    return cv2.blur(noise_img, (3, 3))


def median_blur(noise_img):
    return cv2.medianBlur(noise_img, 3)


def canny(base_image):
    '''
    Contour search algorithm
    '''

    return cv2.Canny(base_image, 100, 330)


def dilate(canny_img):
    kernel = np.ones((3, 3)) / 9.0

    return cv2.dilate(canny_img, kernel)


def erode(dilated_img):
    '''
    Also known as shrink
    '''
    kernel = np.ones((3, 3)) / 9.0

    return cv2.erode(dilated_img, kernel)


def main_convolution():
    img_1 = cv2.imread("home/home1.png")
    img_2 = cv2.imread("home/home2.png")

    noise_img = img_2 - img_1
    canny_img = canny(img_2)
    dilated_img = dilate(canny_img)

    # plt.imshow(cv2.cvtColor(median_blur(noise_img), cv2.COLOR_BGR2RGB))
    plt.imshow(erode(dilated_img), cmap="gray")

    plt.show()


def create_histogram(base_img, color_idx):
    return cv2.calcHist([base_img], [color_idx], None, [256], [0, 256])


def equalize(g_img):
    return cv2.equalizeHist(g_img)


def grayscale_img(base_img):
    return cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)


def create_clahe(g_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    return clahe.apply(g_img)


def adjust_brightness(g_img, is_clahe: bool):
    if is_clahe:
        return create_clahe(g_img)
    else:
        return equalize(g_img)


def two_dim_hist(hsv_img):
    return cv2.calcHist([hsv_img], [0, 1], None, [22, 32], [0, 180, 0, 256])


def main_histogram():
    tree_img = cv2.imread("tree.jpg")

    g_img = grayscale_img(tree_img)

    eq_img = adjust_brightness(g_img, False)
    cl_img = adjust_brightness(g_img, True)

    hsv_img = cv2.cvtColor(tree_img, cv2.COLOR_BGR2HSV)

    plt.imshow(two_dim_hist(hsv_img), cmap="gray")

    plt.show()


main_histogram()
