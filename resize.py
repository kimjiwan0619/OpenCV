import cv2
img = cv2.imread('./data/house.jpg')
# 축소
shrink = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

#shrink = cv2.resize(img, (540, 420), 0, 0, interpolation=cv2.INTER_LINEAR)
cv2.imwrite('./data/house.jpg', shrink)