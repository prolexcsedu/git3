import cv2
import numpy as np

RECT_MIN_WIDTH = 80
RECT_MIN_HEIGHT = 80
LINE_POSITION_Y = 550
OFFSET = 6
LINE_IN_X_START = 25
LINE_IN_X_END = 600
LINE_OUT_X_START = 600
LINE_OUT_X_END = 1200

center_coordinates = []
vehicles_in = 0
vehicles_out = 0

def get_center(x, y, w, h):
    return x + int(w / 2), y + int(h / 2)

cap = cv2.VideoCapture('video.mp4')

algorithm = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)
    fg_mask = algorithm.apply(blur)
    dilated = cv2.dilate(fg_mask, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame, (LINE_IN_X_START, LINE_POSITION_Y), (LINE_IN_X_END, LINE_POSITION_Y), (0, 0, 255), 2)
    cv2.line(frame, (LINE_OUT_X_START, LINE_POSITION_Y), (LINE_OUT_X_END, LINE_POSITION_Y), (0, 255, 0), 2)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= RECT_MIN_WIDTH and h >= RECT_MIN_HEIGHT:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            center = get_center(x, y, w, h)
            center_coordinates.append(center)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

            if LINE_IN_X_START < center[0] < LINE_IN_X_END and LINE_POSITION_Y - OFFSET < center[1] < LINE_POSITION_Y + OFFSET:
                vehicles_in += 1
                center_coordinates.remove(center)
            elif LINE_OUT_X_START < center[0] < LINE_OUT_X_END and LINE_POSITION_Y - OFFSET < center[1] < LINE_POSITION_Y + OFFSET:
                vehicles_out += 1
                center_coordinates.remove(center)

    cv2.putText(frame, "In : " + str(vehicles_in), (400, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(frame, "Out : " + str(vehicles_out), (800, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    cv2.imshow("Foreground Mask", dilated)
    cv2.imshow("Vehicle Counter", frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
