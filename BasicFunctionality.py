import numpy as np
import cv2
import time

"""
height, width,channel = img1.shape
print(height, width , channel)

height, width, channel = img2.shape
print(height, width , channel)"""
#Absolute difference

"""
img1 = cv2.imread('TestImg3.png')
img2 = cv2.imread('TestImg4.png')

diff_frame = cv2.absdiff(img1, img2)
cv2.imshow("diff_frame", diff_frame)
print('img1')
print(img1)
print('img2')
print(img2)
print('diff')
print(diff_frame)
cv2.waitKey(0)
"""

#Arithmetic addition
"""
img1 = cv2.imread('flower1.jpg')
img2 = cv2.imread('flower1.jpg')

height, width, channel = img1.shape
img3 = np.zeros((height, width, channel), np.uint8)
img3 = img1 - img2
cv2.imshow('img3', img3)
cv2.waitKey(0)
print(img3)
print('img4')

cv2.subtract(img1,img2,img3)
cv2.imshow('img4', img3)
print(img3)
cv2.waitKey(0)
w = 50
dst = cv2.addWeighted(img1, float(100 - w) * 0.01, img2, float(w) * 0.01, 0)
"""


#Arithmetic subtraction
"""
img1 = cv2.imread('flower1.jpg')
img2 = cv2.imread('flower2.jpg')

height, width, channel = img1.shape
img3 = np.zeros((height, width, channel), np.uint8)
img3 = img1 - img2
cv2.imshow("img3", img3)
cv2.waitKey(0)
"""
#Bitwise: AND, OR, XOR, NOT

"""
img1 = cv2.imread('TestImg1.png')
img2 = cv2.imread('TestImg2.png')

b1, g1, r1 = cv2.split(img1)
b2, g2, r2 = cv2.split(img2)

bit_and = cv2.bitwise_and(img1, img2)
bit_or = cv2.bitwise_or(img1, img2)
bit_not1 = cv2.bitwise_not(img1)
bit_not2 = cv2.bitwise_not(img2)
bit_xor = cv2.bitwise_xor(img1, img2)
b, g, r = cv2.split(bit_and)
cv2.imshow("bit_and", bit_and)
cv2.waitKey(0)
cv2.imshow("bit_or", bit_or)
cv2.waitKey(0)
cv2.imshow("bit_not1", bit_not1)
cv2.waitKey(0)
cv2.imshow("bit_not2", bit_not2)
cv2.waitKey(0)
cv2.imshow("bit_xor", bit_xor)
cv2.waitKey(0)
"""

#Pixel-wise multiplication
"""
img1 = cv2.imread('./data/flower1.jpg')
h, w, c = img1.shape
img5 = np.full((h, w, c), 4, dtype='uint8')
img3 = img1*img5
print(img3)
print('-------------------------------------')
print(img3)
cv2.imshow('img1*img2', img3)
cv2.waitKey(0)
cv2.multiply(img1, img5, img3)
cv2.imshow('img1*img2', img3)
cv2.waitKey(0)
"""

#Channel combine, Channel extract

"""
img1 = cv2.imread('flower1.jpg')
b, g, r = cv2.split(img1) # extract b, g, r from img1
cv2.imshow("img1r", r)
print(r)
cv2.waitKey(0)

# black img
height, width, channel = img1.shape
img2 = np.zeros((height, width, channel), np.uint8)
b2, g2, r2 = cv2.split(img2)

# merge r in img1 with black img
r = cv2.merge((b2, g2, r))
cv2.imshow("img1r", r)
print('-------------------------------------')
print(r)
cv2.waitKey(0)
"""

#Color convert
"""
img1 = cv2.imread('flower1.jpg')
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # convert image to gray

cv2.imshow('gray', gray)
cv2.waitKey(0)
"""

#Convert bit depth

"""
img1 = cv2.imread('flower1.jpg')
img1 = img1.astype('uint16')
img8 = (img1/256).astype('uint8')
print(img1)
print('----------------------------------------')
print(img8)
"""
#Table lookup


img1 = cv2.imread('./data/flower1.jpg')
brightness_factor = 2
start = time.time()
table = np.array([ i*brightness_factor for i in range (0,256)]).clip(0,255).astype('uint8')
img2 = cv2.LUT(img1, table)
print(time.time() - start)
cv2.imshow('brighter', img2)
cv2.waitKey(0)

h,w,c = img1.shape
start = time.time()
img3 = np.full((h, w, c), 2, dtype='uint8')
img4 = cv2.multiply(img1, img3)
print(time.time() - start)

cv2.destroyAllWindows()
