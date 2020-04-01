import cv2
img = cv2.imread('./data/building.jpg')
# 축소
shrink = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

#shrink = cv2.resize(img, (540, 420), 0, 0, interpolation=cv2.INTER_LINEAR)
cv2.imwrite('./data/building.jpg', shrink)