'''haar face detector'''

import cv2

face_cascade = cv2.CascadeClassifier(
    'haarCascades/haarcascade_frontalface_default.xml')

gray = cv2.imread('2.ppm', cv2.IMREAD_GRAYSCALE)


faces = face_cascade.detectMultiScale(gray, 1.1, 5)
for (x, y, w, h) in faces:
    gray = cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 2)


cv2.imshow('img', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
