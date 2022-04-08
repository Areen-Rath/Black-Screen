import time
import numpy as np
import cv2
import keyboard

fourcc = cv2.VideoWriter_fourcc(*"XVID")
output = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))

cap = cv2.VideoCapture(0)
time.sleep(2)

image = 0
for i in range(60):
    ret, image = cap.read()
image = cv2.resize(image, (640, 480))
image = np.flip(image, axis = 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    frame = np.flip(frame, axis = 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    l_black = np.array([0, 0, 0])
    u_black = np.array([50, 50, 50])

    mask1 = cv2.inRange(hsv, l_black, u_black)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    mask2 = cv2.bitwise_not(mask1)

    result1 = cv2.bitwise_and(frame, frame, mask = mask2)
    result2 = cv2.bitwise_and(image, image, mask = mask1)

    final_output = cv2.addWeighted(result1, 1, result2, 1, 0)
    output.write(final_output)

    cv2.imshow("Magic", final_output)
    cv2.waitKey(1)

    if keyboard.is_pressed("q") or keyboard.is_pressed("esc"):
        break

cap.release()
cv2.destroyAllWindows()