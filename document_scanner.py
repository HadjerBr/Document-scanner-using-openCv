# Importing libraries:
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# reading the original document:
document = cv.imread("data/document.jpg")

# converting to gray scale:
gray = cv.cvtColor(document, cv.COLOR_BGR2GRAY)

# Gaussian blurring:
blur = cv.GaussianBlur(gray, (3, 3), 0)

# Canny edge detection:
canny = cv.Canny(blur, 40, 200)

# Thresholding:
ret, thr = cv.threshold(canny, 127, 255, cv.THRESH_BINARY)

# Contour finding and drawing:
contours, _ = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv.contourArea, reverse=True)[:1]
contour_drawn_document = document.copy()
cv.drawContours(contour_drawn_document, contours, 0, (0, 255, 0), 2)

# Detecting Corners and getting the contour dimensions:
p = cv.arcLength(contours[0], True)
approx = cv.approxPolyDP(contours[0], 0.01*p, True)
x, y, w, h = cv.boundingRect(approx)
if len(approx) == 4:
    # getting the 4 corners coordinates:
    points = np.float32(approx)
else:
    print("Cannot detect the document contours.")
    exit()
    
# Perspective Transform:
width, height = w, h
if width < 200 or height < 200:
    width, height = 300, 500
destination_points = np.float32([[0, 0], [0, height], [width, height], [width, 0] ])
transform_matrix = cv.getPerspectiveTransform(points, destination_points)
warped_image = cv.warpPerspective(contour_drawn_document, transform_matrix, (width, height))

# Matplotlib presentation:
images = [document, canny, contour_drawn_document, warped_image]
titles = ["Original Document", "Canny Edge Detection", "Contour", "Scanned Document"]

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()


