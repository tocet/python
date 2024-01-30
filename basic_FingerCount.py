import cv2
import os
import handtrackingclass as htc

wCam, hCam = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

fingerPath = "FingerImages"
myList = os.listdir(fingerPath)
#print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{fingerPath}/{imPath}')
    overlayList.append(image)

detector = htc.handDetector(detectonConfidence=0.8)

# fingers: index, middle, ring, pinkey
fingertipsId = [8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHand(img,draw = True)
    landmarkList = detector.findPostition(img,draw = False)
    print(landmarkList)

    if len(landmarkList) != 0:
        fingers = []
        #thumb
        if landmarkList[4][1] > landmarkList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # fingers: index, middle, ring, pinky
        for id in range(0,4):
            if landmarkList[fingertipsId[id]][2] < landmarkList[fingertipsId[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        print(fingers)
        countFingers = fingers.count(1)
        #print(countFingers)
        h,w,c = overlayList[countFingers].shape
        img[0:h, 0:w] = overlayList[countFingers]
        cv2.putText(img,str(countFingers),(20,280),cv2.FONT_HERSHEY_COMPLEX,2,(255,170,0),3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)