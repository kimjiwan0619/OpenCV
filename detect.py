import cv2
import numpy as np

cap = cv2.VideoCapture('vtest.avi')
if (not cap.isOpened()):
    print('Error opening video')

height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

#동영상 저장하기
fourcc = cv2.VideoWriter_fourcc(*'WMV2')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (width, height))

acc_gray = np.zeros(shape=(height, width), dtype=np.float32)
acc_bgr = np.zeros(shape=(height, width, 3), dtype=np.float32)
t = 0

while (cap.isOpened()):
    retval, frame = cap.read()
    if not retval:
        break
    t += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.accumulate(gray, acc_gray)
    avg_gray = acc_gray / t
    dst_gray = cv2.convertScaleAbs(avg_gray)

    cv2.accumulate(frame, acc_bgr)
    avg_bgr = acc_bgr / t
    dst_bgr = cv2.convertScaleAbs(avg_bgr)

    out.write(dst_bgr)

    cv2.imshow('frame', dst_gray)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()