import cv2
import mediapipe as mp

class handDetector():
    def __init__(self, mode = False, maxHands = 2, modelComplexity = 1, detectonConfidence = 0.5, trackingConfidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectonConfidence = detectonConfidence
        self.trackingConfidence = trackingConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelComplexity,
                                        self.detectonConfidence,self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        # fingers: index, middle, ring, pinky
        self.fingerTipsId = [8, 12, 16, 20]

    def findHand(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_landmarks,self.mpHands.HAND_CONNECTIONS)
        return img

    def findPostition(self, img, handNumber = 0, draw = True):
        self.landmarkList = []
        if self.results.multi_hand_landmarks:
            processedHand = self.results.multi_hand_landmarks[handNumber]
            for id, lm in enumerate(processedHand.landmark):
                height, width, c = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                self.landmarkList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 255, 0), cv2.FILLED)
        return self.landmarkList