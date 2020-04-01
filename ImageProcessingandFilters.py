import cv2
import numpy as np
from scipy.spatial import distance as dist
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

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
"""
#blending

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

# Contour 영역 내에 텍스트 쓰기
"""
def setLabel(image, str, contour):
   fontface = cv2.FONT_HERSHEY_SIMPLEX
   scale = 0.6
   thickness = 2
   size = cv2.getTextSize(str, fontface, scale, thickness)
   text_width = size[0][0]
   text_height = size[0][1]
   x, y, width, height = cv2.boundingRect(contour)
   pt = (x + int((width - text_width) / 2), y + int((height + text_height) / 2))
   cv2.putText(image, str, pt, fontface, scale, (255, 255, 255), thickness, 8)
# 컨투어 내부의 색을 평균내서 red, green, blue 중 어느 색인지 체크
def label(image, contour):
   mask = np.zeros(image.shape[:2], dtype="uint8")
   cv2.drawContours(mask, [contour], -1, 255, -1)
   mask = cv2.erode(mask, None, iterations=2)
   mean = cv2.mean(image, mask=mask)[:3]
   minDist = (np.inf, None)
   for (i, row) in enumerate(lab):
       d = dist.euclidean(row[0], mean)
       if d < minDist[0]:
           minDist = (d, i)
   return colorNames[minDist[1]]
# 인식할 색 입력
colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]
colorNames = ["red", "green", "blue"]
lab = np.zeros((len(colors), 1, 3), dtype="uint8")
for i in range(len(colors)):
   lab[i] = colors[i]
lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB)
# 원본 이미지 불러오기
image = cv2.imread("./data/color.jpg", 1)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
# 이진화
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
# 색검출할 색공간으로 LAB사용
img_lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
thresh = cv2.erode(thresh, None, iterations=2)
cv2.imshow("Thresh", thresh)
# 컨투어 검출
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 컨투어 리스트가 OpenCV 버전에 따라 차이있기 때문에 추가
if len(contours) == 2:
   contours = contours[0]
elif len(contours) == 3:
   contours = contours[1]

# 컨투어 별로 체크
for contour in contours:
   cv2.imshow("Image", image)
   cv2.waitKey(0)
   # 컨투어를 그림
   cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
   # 컨투어 내부에 검출된 색을 표시
   color_text = label(img_lab, contour)
   setLabel(image, color_text, contour)

cv2.imshow("Image", image)
cv2.waitKey(0)
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
scharrx= cv2.Scharr(img, -1, 1, 0)
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
cv2.destroyAllWindows()


