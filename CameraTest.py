import cv2

wCam,hCam = 800,600
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

while True:
    success, img = cap.read()
    cv2.imshow("Image",img)
    cv2.waitKey()