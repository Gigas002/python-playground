# %% imports and initialization of image object

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

tokyo_tower = Image.open("tokyo_tower.png")

# %% image modifications

gray_tower = tokyo_tower.convert("L")
gray_tower.save("tokyo_tower_gray.png")

tokyo_tower_resized = tokyo_tower.resize((250, 375))
tokyo_tower_rotated = tokyo_tower.rotate(60, expand=True)
tokyo_tower_translated = tokyo_tower.rotate(0, translate=(100, 100))

# plt.imshow(gray_tower, cmap="gray")
# plt.imshow(tokyo_tower_translated)
# plt.show()

# %% convert to numpy object

color_image = np.array(tokyo_tower)
# print(type(color_image))
print(color_image.shape)
# for 0;0 pixel get color info
# color is an RGB uint8 array, where R's index is 0, G = 1 and B = 2
print(color_image[0, 0, :])

# %% image color as histogram

# histogram for all colors
# hist_c, bins_c = np.histogram(color_image.flatten(), bins= 256 )

# histograms for each color
# hist_r, bins_r = np.histogram(color_image[:, :, 0].flatten(), bins=256)
# plt.subplot(1, 3, 1)
# plt.plot(hist_r)
# hist_g, bins_g = np.histogram(color_image[:, :, 1].flatten(), bins=256)
# plt.subplot(1, 3, 2)
# plt.plot(hist_g)
# hist_b, bins_b = np.histogram(color_image[:, :, 0].flatten(), bins=256)
# plt.subplot(1, 3, 3)
# plt.plot(hist_b)

# plt.show()

#%% Convert numpy array to image

# 1. Read from converted uint8 array
color_image = np.array(tokyo_tower)
color_pil = Image.fromarray(np.uint8(color_image))
# plt.imshow(color_pil)

# 2. Use array as is

# Modify image
half_image = color_image // 2

plt.imshow(half_image)

# Show data

# plt.show()

# %% Rocket fun

rocket_org = Image.open("H2A.jpg")

rocket_crop_list = []
for i in range(4):
    rocket_crop_list.append(rocket_org.crop((0, 200 * i, rocket_org.width, 200 * (i + 1))))

rocket_crop_list[0].save("rocket_anim.gif",
                         save_all = True,
                         append_images = rocket_crop_list[1:4],
                         loop = 0, duration = 500)

rocket_org_array = np.array(rocket_org)

# monochromatic red only 
rocket_r = np.zeros((rocket_org.height, rocket_org.width, 3), dtype=np.uint8)
rocket_r[:, :, 0] = rocket_org_array[:, :, 0]

# Monochromatic green only 
rocket_g = np.zeros((rocket_org.height, rocket_org.width, 3), dtype=np.uint8)
rocket_g[:, :, 1] = rocket_org_array[:, :, 1]

# monochromatic blue only 
rocket_b = np.zeros((rocket_org.height, rocket_org.width, 3), dtype=np.uint8)
rocket_b[:, :, 2] = rocket_org_array[:, :, 2]

# Display monochromatic images side by side 
rocket_monos = Image.new("RGB", (rocket_org.width * 3, rocket_org.height))
rocket_monos.paste(Image.fromarray(rocket_r), (0, 0))
rocket_monos.paste(Image.fromarray(rocket_g), (rocket_org.width, 0))
rocket_monos.paste(Image.fromarray(rocket_b), (rocket_org.width * 2, 0))

# plt.imshow(rocket_monos)
# plt.imshow(Image.open("rocket_anim.gif"))

# Make a duplicate so as not to change the original (rocket_r) of the red-only image data
rocket_r_trimmed = np.copy(rocket_r)

# binarize red color based on color intensity 240 
rocket_r_wb = rocket_r_trimmed[:, :, 0] // 240

print(rocket_r_trimmed.shape)

# Multiplication of red-only image data and binarized data 
rocket_r_trimmed[:, :, 0] *= rocket_r_wb

# display image data leaving only pixels with red intensity greater than 240
plt.imshow(rocket_r_trimmed)

# plt.show()

# %%
