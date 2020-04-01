import cv2
import numpy as np

cap = cv2.VideoCapture('./data/vtest.avi')
if (not cap.isOpened()):
    print('Error opening video')

height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

acc_gray = np.zeros(shape=(height, width), dtype=np.float32)
acc_bgr = np.zeros(shape=(height, width, 3), dtype=np.float32)
t = 0

while (cap.isOpened()):
    retval, frame = cap.read()
    if not retval:
        break
    t += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img_canny = cv2.Canny(gray, 100, 200)
    cv2.imshow("Canny frame", img_canny)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()