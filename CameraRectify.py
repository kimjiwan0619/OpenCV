import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt

filepath = ''
criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_cal = (cv2.TERM_CRITERIA_EPS +
                     cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints_l = []  # 2d points in image plane.
imgpoints_r = []  # 2d points in image plane.

cal_path = filepath

images_right = glob.glob(cal_path + 'RIGHT/*.JPG')
images_left = glob.glob(cal_path + 'LEFT/*.JPG')
images_left.sort()
images_right.sort()

for i, fname in enumerate(images_right):
    img_l = cv2.imread(images_left[i])
    img_r = cv2.imread(images_right[i])

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None)

    # If found, add object points, image points (after refining them)
    objpoints.append(objp)

    if ret_l is True:
        rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                              (-1, -1), criteria)
        imgpoints_l.append(corners_l)

        # Draw and display the corners
        ret_l = cv2.drawChessboardCorners(img_l, (9, 6),
                                          corners_l, ret_l)
        #cv2.imshow(images_left[i], img_l)
        #cv2.waitKey(500)

        if ret_r is True:
            rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                  (-1, -1), criteria)
            imgpoints_r.append(corners_r)

            # Draw and display the corners
            ret_r = cv2.drawChessboardCorners(img_r, (9, 6),
                                              corners_r, ret_r)
            #cv2.imshow(images_right[i], img_r)
            #cv2.waitKey(500)
        img_shape = gray_l.shape[::-1]
        dims = img_shape
    rt, M1, d1, r1, t1 = cv2.calibrateCamera(
        objpoints, imgpoints_l, img_shape, None, None)
    rt, M2, d2, r2, t2 = cv2.calibrateCamera(
        objpoints, imgpoints_r, img_shape, None, None)

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
flags |= cv2.CALIB_USE_INTRINSIC_GUESS
flags |= cv2.CALIB_FIX_FOCAL_LENGTH
# flags |= cv2.CALIB_FIX_ASPECT_RATIO
flags |= cv2.CALIB_ZERO_TANGENT_DIST
# flags |= cv2.CALIB_RATIONAL_MODEL
# flags |= cv2.CALIB_SAME_FOCAL_LENGTH
# flags |= cv2.CALIB_FIX_K3
# flags |= cv2.CALIB_FIX_K4
# flags |= cv2.CALIB_FIX_K5

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                        cv2.TERM_CRITERIA_EPS, 100, 1e-5)
ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l,
    imgpoints_r, M1, d1, M2,
    d2, dims,
    criteria=stereocalib_criteria, flags=flags)

print('Intrinsic_mtx_1', M1)
print('dist_1', d1)
print('Intrinsic_mtx_2', M2)
print('dist_2', d2)
print('R', R)
print('T', T)
print('E', E)
print('F', F)


# for i in range(len(self.r1)):
#     print("--- pose[", i+1, "] ---")
#     self.ext1, _ = cv2.Rodrigues(self.r1[i])
#     self.ext2, _ = cv2.Rodrigues(self.r2[i])
#     print('Ext1', self.ext1)
#     print('Ext2', self.ext2)

print('')

camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                    ('dist2', d2), ('rvecs1', r1),
                    ('rvecs2', r2), ('R', R), ('T', T),
                    ('E', E), ('F', F)])


R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(M1, d1, M2, d2, dims, R, T)
print('R1',R1)
print('R2',R2)
print('P1',P1)
print('P2',P2)
print('Q',Q)

img = cv2.imread('./LEFT/left12.jpg')
img2 = cv2.imread('./RIGHT/right12.jpg')

h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(M1, d1, (w, h), 1, (w, h))

mapx1, mapy1 = cv2.initUndistortRectifyMap(M1, d1, R1, P1, dims, cv2.CV_32F)
mapx2, mapy2 = cv2.initUndistortRectifyMap(M2, d2, R2, P2, dims, cv2.CV_32F)

dst = cv2.remap(img, mapx1, mapy1, cv2.INTER_LINEAR)
dst2 = cv2.remap(img2, mapx2, mapy2, cv2.INTER_LINEAR)
"""
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
dst2 = dst2[y:y+h, x:x+w]
"""
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
x, y, w, h = roi
dst2 = dst2[y:y + h, x:x + w]
cv2.imwrite('Rectifyl.png', dst)
cv2.imwrite('Rectifyr.png', dst2)

