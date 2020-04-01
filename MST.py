import cv2
import numpy as np

roi = None
drag_start = None
mouse_status = 0
tracking_start = False


def onMouse(event, x, y, flags, param=None):
    global roi
    global drag_start
    global mouse_status
    global tracking_start

    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (x, y)
        roi = (0, 0, 0, 0)  # ROI를 재설정하는 경우를 위한 초기화
        tracking_start = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if flags == cv2.EVENT_FLAG_LBUTTON:
            xmin = min(x, drag_start[0])
            ymin = min(y, drag_start[1])
            xmax = max(x, drag_start[0])
            ymax = max(y, drag_start[1])
            roi = (xmin, ymin, xmax, ymax)
            mouse_status = 1
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_status = 2


cv2.namedWindow('meanShift tracking')
cv2.setMouseCallback('meanShift tracking', onMouse)

cap = cv2.VideoCapture('./data/vtest.avi')
if (not cap.isOpened()):
    print('Error opening video')
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
roi_mask = np.zeros((height, width), dtype=np.uint8)
term_crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1)

fourcc = cv2.VideoWriter_fourcc(*'WMV2')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (width, height))

while True:
    ret, meanframe = cap.read()
    if not ret:
        break

    camframe = meanframe.copy()
    hsv = cv2.cvtColor(meanframe, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0., 60., 32.), (180., 255., 255.))
    # Checks if array elements lie between the elements of two other arrays.

    if mouse_status == 1:
        x1, y1, x2, y2 = roi
        cv2.rectangle(meanframe, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if mouse_status == 2:
        print('Initializing...', end=' ')
        mouse_status = 0
        x1, y1, x2, y2 = roi
        if (np.abs(x1 - x2) < 10) or (np.abs(y1 - y2) < 10):
            print('failed. Too small ROI. (Width: %d, Height: %d)' % (np.abs(x1 - x2), np.abs(y1 - y2)))
            continue

        mask_roi = mask[y1:y2, x1:x2]
        hsv_roi = hsv[y1:y2, x1:x2]

        hist_roi = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])
        cv2.normalize(hist_roi, hist_roi, 0, 255, cv2.NORM_MINMAX)
        track_window1 = (x1, y1, x2 - x1, y2 - y1)
        track_window2 = (x1, y1, x2 - x1, y2 - y1)
        tracking_start = True
        print('Done.')

    if tracking_start:
        backP = cv2.calcBackProject([hsv], [0], hist_roi, [0, 180], 1)
        # Calculates the back projection of a histogram.
        backP &= mask
        cv2.imshow('backP', backP)

        ret, track_window1 = cv2.meanShift(backP, track_window1, term_crit)
        # Finds an object on a back projection image.
        x, y, w, h = track_window1
        cv2.rectangle(meanframe, (x, y), (x + w, y + h), (0, 0, 255), 2)

        track_box, track_window2 = cv2.CamShift(backP, track_window2, term_crit)
        # Finds an object center, size, and orientation.
        x, y, w, h = track_window2
        cv2.rectangle(camframe, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.ellipse(camframe, track_box, (0, 255, 255), 2)
        pts = cv2.boxPoints(track_box)  # Finds the four vertices of a rotated rect.
        pts = np.int0(pts)
        dst = cv2.polylines(camframe, [pts], True, (0, 0, 255), 2)

    cv2.imshow('meanShift tracking', meanframe)
    out.write(meanframe)
    cv2.imshow('CamShift tracking', camframe)
    key = cv2.waitKey(0)
    if key == 27:
        break

if cap.isOpened():
    cap.release()
    out.release()
cv2.destroyAllWindows()
