import cv2
dst = cv2.imread('./data/tsukuba_l.png')
dst2 = cv2.imread('./data/tsukuba_r.png')

dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2GRAY)
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

disparity = stereo.compute(dst, dst2, disparity=cv2.CV_32F)
norm_coeff = 255 / disparity.max()
cv2.imshow("disparity", disparity * norm_coeff / 255)
cv2.waitKey(0)