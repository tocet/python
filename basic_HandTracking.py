#example - base of HandTrackingClass
import cv2
import mediapipe as mp
import time

c_time = 0
p_time = 0

#webcam
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    sucess, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
           mpDraw.draw_landmarks(img,hand_landmarks,mpHands.HAND_CONNECTIONS)
           for id, lm in enumerate(hand_landmarks.landmark):
                print(id,lm)
                height,width,c = img.shape
                cx, cy = int(lm.x*width), int(lm.y*height)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img,(cx,cy),15,(255,255,0),cv2.FILLED)

#FPS calculation
    c_time = time.time()
    fps = int(1/(c_time - p_time))
    p_time = c_time
    cv2.putText(img,str(fps),(15,80),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),2)
    cv2.imshow("Image",img)
    cv2.waitKey(1)