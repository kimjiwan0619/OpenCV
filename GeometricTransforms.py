import cv2
import numpy as np
from matplotlib import pyplot as plt
#Scale/Resize
#cv2.resize(img, dsize, fx, fy, interpolation) 보간법

img = cv2.imread('zebra.jpg')
# 축소
shrink = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
#height, width = shrink.shape[:2]
# Manual Size지정
zoom1 = cv2.resize(shrink, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)

# 배수 Size지정
zoom2 = cv2.resize(shrink, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
zoom3 = cv2.resize(shrink, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
zoom4 = cv2.resize(shrink, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
zoom5 = cv2.resize(shrink, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)


cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.imshow('INTER_AREA', zoom1)
cv2.waitKey(0)
cv2.imshow('INTER_NEAREST', zoom2)
cv2.waitKey(0)
cv2.imshow('INTER_LINEAR', zoom3)
cv2.waitKey(0)
cv2.imshow('INTER_CUBIC', zoom4)
cv2.waitKey(0)
cv2.imshow('INTER_LANCZOS4', zoom5)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Warp Affine

"""
img = cv2.imread('Checkboard.png')
rows, cols, ch = img.shape

pts1 = np.float32([[100,50],[200,50],[100,100]])

pts2 = np.float32([[100,75],[200,25],[100,125]])

# pts1의 좌표에 표시. Affine 변환 후 이동 점 확인.
cv2.circle(img, (100,50), 10, (255,0,0),-1)
cv2.circle(img, (200,50), 10, (0,255,0),-1)
cv2.circle(img, (100,100), 10, (0,0,255),-1)

M = cv2.getAffineTransform(pts1, pts2)

dst = cv2.warpAffine(img, M, (cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('image')
plt.subplot(122),plt.imshow(dst),plt.title('Affine')
plt.show()
"""