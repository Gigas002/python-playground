import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def concat_images(base_image, concat_image):
    # vconcat for vertical concat
    # hconcat for horizontal concat
    return cv2.vconcat([base_image, concat_image])

def crop_image_by_coordinates(base_image, vmin, vmax, hmin, hmax):
    return base_image[vmin:vmax + 1, hmin:hmax + 1, :]

def slice_image(base_image : np.ndarray, parts = 4):
    sliced = []

    h, w, c = base_image.shape

    height = h // 4
    width = w // 4

    # in this example we need vertical-only slice
    for i in range(parts):
        sliced.append(base_image[(height * i):(height * (i + 1)), :, :])
        # sliced.append(base_image[(height * i):(height * (i + 1)),
        #                          (width * i):(width * (i + 1)), :])

    return sliced

def create_animation(images):
    # This will result in a GIF image with 800px width and 250px height
    # figsize=(horizontal inch, vertical inch), dpi=resolution (same as multiplier)
    figure = plt.figure(figsize = (4, 1), dpi = 250)

    # no padding
    ax = figure.add_axes([0, 0, 1, 1])

    # Hide elements such as the axis and title of the graph
    ax.axis('off')

    gif = []
    # Put each quadrant in RGB format into a new list (rocket_crop_gif_list)
    for i in range(4):
        gif.append([ax.imshow(images[i])])

    # create an animation
    return animation.ArtistAnimation(figure, gif, interval = 500, repeat_delay = 500)

def create_monos(rgb_image):
    '''
    Pass RGB image inside
    '''

    h, w, c = rgb_image.shape

    # Red
    img_r = np.zeros((h, w, 3), dtype=np.uint8)
    img_r[:, :, 0] = rgb_image[:, :, 0]

    # Green
    img_g = np.zeros((h, w, 3), dtype=np.uint8)
    img_g[:, :, 1] = rgb_image[:, :, 1]

    # Blue
    img_b = np.zeros((h, w, 3), dtype=np.uint8)
    img_b[:, :, 2] = rgb_image[:, :, 2]

    # display monochrome images side by side
    return cv2.hconcat([img_r, img_g, img_b])

def remove_blue(rgb_image):
    h, w, c = rgb_image.shape

    img_gr = np.zeros((h, w, 3), dtype=np.uint8)
    img_gr[:, :, 0] = rgb_image[:, :, 0]
    img_gr[:, :, 1] = rgb_image[:, :, 1]

    return img_gr

def binarization(rgb_image, strength = 128):
    grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    return grayscale_image // strength

def red_binarization(rgb_image, strength = 240):
    h, w, c = rgb_image.shape

    img_r = np.zeros((h, w, 3), dtype=np.uint8)
    img_r[:, :, 0] = rgb_image[:, :, 0]

    # make a duplicate of img_r array, because we don't want to change it
    img_r_trimmed = np.copy(img_r)

    # binarization, bazed on red color with strength 240
    # contains "0" and "1", so data either exists, or doesn't exist at all
    img_r_wb = img_r_trimmed[:, :, 0] // strength

    # multiplication of red-only image data and binarized data
    img_r_trimmed[:, :, 0] *= img_r_wb

    return img_r_trimmed, img_r_wb

def masking(base_image, mask_image):
    h, w, c = base_image.shape

    # data for masking
    masking_data = np.zeros((h, w, c), dtype=np.uint8)

    # For all RGBs, binarized values ​​converted to "0" and "255"
    for i in range(c):
        masking_data[:, :, i] = mask_image * 255

    return cv2.bitwise_and(base_image, masking_data)

def rocket_main():
    rocket_org = cv2.cvtColor(cv2.imread("H2A.jpg"), cv2.COLOR_BGR2RGB)

    # anim = create_animation(slice_image(rocket_org))
    # anim.save('rocket_anim_2.gif', writer='pillow')

    # (binarized, mask) = red_binarization(rocket_org)
    # plt.imshow(masking(rocket_org, mask))
    # plt.imshow(binarization(rocket_org), cmap="gray")

    plt.show()

def img_overlay(img_1, img_2):
    return cv2.cvtColor((img_2 - img_1), cv2.COLOR_BGR2RGB)

def alpha_blending(img_1, img_2):
    # transparency sum must be 1.0. The lesser the value - the more transparent the image
    return cv2.cvtColor(cv2.addWeighted(img_1, 0.7, img_2, 0.3, 0), cv2.COLOR_BGR2RGB)

def ball_main():
    img_1 = cv2.imread("home/home1.png")
    img_2 = cv2.imread("home/home2.png")

    plt.imshow(alpha_blending(img_1, img_2))
    plt.show()

ball_main()
