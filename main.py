import timeit, cv2
from matplotlib import pyplot

IMAGE_FILENAME = 'text_withnikud_1.png'
start_timer = timeit.default_timer()

img = cv2.imread(IMAGE_FILENAME, cv2.IMREAD_GRAYSCALE)

img_smoothed = cv2.medianBlur(img, 5)

ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

height, width = img.shape[:2]
print("height = " + str(height) + ", width = " + str(width))

titles = ['Original Image', 'Smoothed', 'Binary Threshold (v=127)', 'Adaptive Mean', 'Adaptive Gaussian']
images = [img, img_smoothed, th1, th2, th3]

for i in range(len(titles)):
    pyplot.subplot(5, 1, i+1)
    pyplot.imshow(images[i], 'gray')
    pyplot.title(titles[i])
    pyplot.xticks([])
    pyplot.yticks([])

stopped_timer = timeit.default_timer()

# cv2.namedWindow('image')
# cv2.imshow('image', img)
pyplot.show()

print("Program duration: " + str(round(stopped_timer - start_timer, 2)) + "s")

k = cv2.waitKey(0)
cv2.destroyAllWindows()
