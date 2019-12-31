import timeit
from matplotlib import pyplot as plt
from helper_funcs import *


IMAGE_FILENAME = 'image1.png'
start_timer = timeit.default_timer()

org_img = cv2.imread(IMAGE_FILENAME, cv2.IMREAD_GRAYSCALE)
height, width = org_img.shape[:2]
print("Image: " + str(IMAGE_FILENAME) + "\nwidth = " + str(width) + ", height = " + str(height))

# Apply Threshold
ret, th_img = cv2.threshold(copy.copy(org_img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
working_img = copy.copy(th_img)

# Switch to white over black
working_img = cv2.bitwise_not(working_img)

# Opening to remove some noise
struct_elem = np.ones((2, 2), np.uint8)
opening_img = cv2.morphologyEx(working_img, cv2.MORPH_OPEN, struct_elem)
working_img = copy.copy(opening_img)

# Contours
contours_image = copy.copy(working_img)
contours, hierarchy = cv2.findContours(contours_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours_image = cv2.drawContours(contours_image, contours, -1, (127, 127, 127), 1)

print("Number of objects in image (contours): " + str(len(contours[0])))

# Debugging:
# titles = ['Original Image', 'Threshold', 'Opening to remove noise', 'Contours on image']
# images = [org_img, th_img, cv2.bitwise_not(opening_img), cv2.bitwise_not(contours_image)]
# for i in range(len(titles)):
#     plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i]), plt.xticks([]), plt.yticks([])
# plt.show()

# Removing demarcation by taking the smallest contour and building kernel according to it
no_demarcation_img = remove_demarcation(contours[0], working_img)

# Debugging:
# plt.subplot(1, 2, 1), plt.imshow(th_img, 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(1, 2, 2), plt.imshow(cv2.bitwise_not(no_demarcation_img), 'gray')
# plt.title('Removed demarcation (not done yet)'), plt.xticks([]), plt.yticks([])
#
# plt.show()

# Return to original image, with only "good" relevent contours
final_img = rebuild_org_img(th_img, contours, no_demarcation_img)

stopped_timer = timeit.default_timer()
print("Program duration: " + str(round(stopped_timer - start_timer, 2)) + "s")

# Printing the final result
plt.subplot(1, 2, 1), plt.imshow(th_img, 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(final_img, 'gray')
plt.title('Final Image'), plt.xticks([]), plt.yticks([])
plt.show()
