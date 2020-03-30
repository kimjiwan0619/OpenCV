import numpy as np
import cv2
import glob

# termination criteria EPS,ITER둘중하나 만족하면 중단 30반복 or 정확도 0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 9, 3), np.float32)
print(objp)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
print('reshape')
print(objp)
# Arrays to store object points and image points from all the images.

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.



images = glob.glob('./data/*.jpg') #all jpg

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print(fname)
        #cv2.imshow('KnV1', img)
       # cv2.waitKey(0)
        objpoints.append(objp)
        # 정교화함
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        #cv2.drawChessboardCorners(gray, (9, 6), corners, ret)
        #cv2.imshow('KnV2', gray)
        #cv2.waitKey(0)
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        cv2.imshow('KnV3', img)
        cv2.waitKey(0)

        # 카메라 메트릭스 왜곡 계수, 회전/이동 벡터
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 카메라 메트릭스 개선
img = cv2.imread('./data/left12.jpg')

h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite('calibresult.png', dst)


#오차계산
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    tot_error += error

print("total error: ", tot_error / len(objpoints))

cv2.destroyAllWindows()