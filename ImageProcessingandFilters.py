import cv2
import numpy as np
from scipy.spatial import distance as dist
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
#Box
"""
img = cv2.imread('./data/Lenna.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img', img)
cv2.waitKey(0)
img_blur = cv2.boxFilter(img, -1, (3,3))
cv2.imshow('boxFilter', img_blur)
cv2.waitKey(0)
"""
#Gaussian
"""
img = cv2.imread('./data/Lenna.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img, (5,5), 0)
cv2.imshow('gaussian',img_blur)
cv2.waitKey(0)

#Median

img = cv2.imread('./data/Lenna.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.medianBlur(img, 5)
cv2.imshow('Median',img_blur)
cv2.waitKey(0)

#Bilateral
img = cv2.imread('./data/Lenna.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.bilateralFilter(img, 5, 250, 10)
cv2.imshow('bilateralFilter',img_blur)
cv2.waitKey(0)
"""
#Equalize Histogram
"""
img = cv2.imread('./data/girl.jpg')
cv2.imshow('src', img)
img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(img)
srcHist = cv2.calcHist(images = [img],
                       channels = [0],
                       mask = None,
                       histSize = [256],
                       ranges = [0, 256])

dstHist = cv2.calcHist(images = [dst],
                       channels = [0],
                       mask = None,
                       histSize = [256],
                       ranges = [0, 256])
cv2.imshow('img', img)
cv2.imshow('dst', dst)
plt.plot(srcHist, color = 'b', label = 'src hist')
plt.plot(dstHist, color = 'r', label = 'dst hist')
plt.legend(loc='best')
plt.show()
cv2.waitKey()
"""
#Dilate, Erode
"""
img = cv2.imread('./data/TestImg9.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = np.ones((5,5), np.uint8)
erode = cv2.erode(img, kernel)
dilation = cv2.dilate(img, kernel)
cv2.imshow('img', img)
cv2.imshow('erode', erode)
cv2.imshow('dilation', dilation)
cv2.waitKey(0)
"""
#Opening Closing
"""
img1 = cv2.imread('./data/TestImg6.png')
cv2.imshow('opening',img1)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('./data/TestImg6.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel)

cv2.imshow('opening',opening)
cv2.imshow('closing',closing)
cv2.waitKey(0)
"""
#Thresholding
"""
img_source = cv2.imread('./data/TestImg5.png',0)

ret,img_result1 = cv2.threshold(img_source, 127, 255, cv2.THRESH_BINARY) # 127값 이상을 255로 변환
ret,img_result2 = cv2.threshold(img_source, 127, 255, cv2.THRESH_BINARY_INV) #반전

cv2.imshow("SOURCE", img_source)
cv2.imshow("THRESH_BINARY", img_result1)
cv2.imshow("THRESH_BINARY_INV", img_result2)

cv2.waitKey(0)
"""

"""
img_source = cv2.imread('./data/TestImg5.png',0)
ret,img_result1 = cv2.threshold(img_source, 127, 255, cv2.THRESH_BINARY)
img_result2 = cv2.adaptiveThreshold(img_source, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
img_result3 = cv2.adaptiveThreshold(img_source, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)

cv2.imshow("SOURCE", img_source)
cv2.imshow("THRESH_BINARY", img_result1)
cv2.imshow("ADAPTIVE_THRESH_MEAN_C", img_result2)
cv2.imshow("ADAPTIVE_THRESH_GAUSSIAN_C", img_result3)

cv2.waitKey(0)
"""
"""
#OTSU는 임계값을 자동으로 계산해줌
img_source = cv2.imread('./data/TestImg6.png', 0)

ret,img_result1 = cv2.threshold(img_source, 127, 255, cv2.THRESH_BINARY)

ret, img_result2 = cv2.threshold(img_source, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

img_blur = cv2.GaussianBlur(img_source, (5,5), 0)
ret, img_result3 = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)



cv2.imshow("SOURCE", img_source)
cv2.imshow("THRESH_BINARY", img_result1)
cv2.imshow("THRESH_OTSU", img_result2)
cv2.imshow("THRESH_OTSU + Gaussian filtering", img_result3)

cv2.waitKey(0)

"""
#Image pyramid

"""
img = cv2.imread('./data/flower1.jpg')

lower_reso = cv2.pyrDown(img) # 원본 이미지의 1/4 사이즈
higher_reso = cv2.pyrUp(img) #원본 이미지의 4배 사이즈

cv2.imshow('img', img)
cv2.imshow('lower', lower_reso)
cv2.imshow('higher', higher_reso)

cv2.waitKey(0)
"""

#blending
"""
A = cv2.imread('./data/apple.jpg')
B = cv2.imread('./data/orange.jpg')

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)

# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpA[i])
    temp = cv2.resize(gpA[i - 1], (GE.shape[:2][1], GE.shape[:2][0]))
    L = cv2.subtract(temp, GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpB[i])
    temp = cv2.resize(gpA[i - 1], (GE.shape[:2][1], GE.shape[:2][0]))
    L = cv2.subtract(temp, GE)
    lpB.append(L)

# Now add left and right halves of images in each level
LS = []
xxx = 0
for la, lb in zip(lpA, lpB):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0:int(cols / 2)], lb[:, int(cols / 2):]))
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
for i in range(1, 6):
    ls_ = cv2.pyrUp(ls_)
    temp = cv2.resize(LS[i], (ls_.shape[:2][1], ls_.shape[:2][0]))
    ls_ = cv2.add(ls_, temp)

# image with direct connecting each half
real = np.hstack((A[:, :int(cols / 2)], B[:, int(cols / 2):]))

b, g, r = cv2.split(A)
A = cv2.merge([r, g, b])
plt.subplot(2, 2, 1), plt.imshow(A), plt.title('A'), plt.xticks([]), plt.yticks([])

b, g, r = cv2.split(B)
B = cv2.merge([r, g, b])
plt.subplot(2, 2, 2), plt.imshow(B), plt.title('B'), plt.xticks([]), plt.yticks([])

b, g, r = cv2.split(ls_)
ls_ = cv2.merge([r, g, b])
plt.subplot(2, 2, 3), plt.imshow(ls_), plt.title('ls_'), plt.xticks([]), plt.yticks([])

b, g, r = cv2.split(real)
real = cv2.merge([r, g, b])
plt.subplot(2, 2, 4), plt.imshow(real), plt.title('real'), plt.xticks([]), plt.yticks([])

plt.show()
"""
#Color Detection
"""
cap = cv2.VideoCapture('./data/vtest.avi')
#동영상 저장하기


height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
fourcc = cv2.VideoWriter_fourcc(*'WMV2')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (width, height))
while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# Red color
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)
    # Blue color
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # Green color
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)

    # Every color except white
    low = np.array([0, 42, 0])
    high = np.array([179, 255, 255])
    mask = cv2.inRange(hsv_frame, low, high)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("Frame", frame)
    cv2.imshow("Red", red)
    cv2.imshow("Blue", blue)
    cv2.imshow("Green", green)
    cv2.imshow("Result", result)
    out.write(red)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
"""
#Integral image

"""
snap = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype='uint8')
print(snap)

intergral_image = cv2.integral( snap )
print(intergral_image)

# if you dont want the top row/left column pad
intergral_image = intergral_image[1:,1:]
print(intergral_image)
"""
#Gradient Magnitude
"""
img = cv2.imread('./data/keyboard.jpg', cv2.IMREAD_GRAYSCALE)

laplacian = cv2.Laplacian(img, -1)
sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)
sobely = cv2.Sobel(img, -1, 0, 1, ksize=3)
scharrx = cv2.Scharr(img, -1, 1, 0)
scharry = cv2.Scharr(img, -1, 0, 1)
cv2.imshow('laplacian',laplacian)
cv2.waitKey(0)
cv2.imshow('sobelx',sobelx)
cv2.waitKey(0)
cv2.imshow('sobely',sobely)
cv2.waitKey(0)
cv2.imshow('scharrx',scharrx)
cv2.waitKey(0)
cv2.imshow('scharry',scharry)
cv2.waitKey(0)
"""
#Histogram

"""
img = cv2.imread('./data/Colorful.jpg')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
"""
"""
img = cv2.imread('./data/flower2.jpg',0)
plt.hist(img.ravel(), 256, [0,256])
plt.show()
"""
#Min/Max Location
"""
img_rgb = cv2.imread('./data/input.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('./data/template2.jpg', 0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

threshold = 0.8
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

cv2.imshow('result', img_rgb)
cv2.waitKey(0)
"""

#Mean & Standard Deviation

"""
img = cv2.imread('./data/flower1.jpg')
color = ('b','g','r')
mean, stddev = cv2.meanStdDev(img)
mean = np.squeeze(mean, axis=1)
print(mean)
plt.bar(np.arange(len(mean)), mean, tick_label = color)
plt.ylim( 0, 200)
plt.show()
"""


