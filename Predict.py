import itertools
import pandas as pd
import pickle
import cv2
import mediapipe as mp
import copy
import numpy as np

with open('Final_test_rc.pkl', 'rb') as f:
    model = pickle.load(f)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 960)

def cal_landmarks(image, landmarks):
    img_width, img_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _,landmark in enumerate(landmarks.landmark):

        landmark_x = min(int(landmark.x * img_width), img_width-1)
        landmark_y = min(int(landmark.y * img_height), img_height-1)
        #print("x {}".format(landmark_x))

        landmark_point.append([landmark_x,landmark_y])

    return landmark_point
def normalizeData(landmark_list):
    list_copy = copy.deepcopy(landmark_list)
    base_x, base_y = 0,0

    for index, landmark_point in enumerate(list_copy):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        list_copy[index][0] = list_copy[index][0] - base_x
        list_copy[index][1] = list_copy[index][1] - base_y

    list_copy = list(itertools.chain.from_iterable(list_copy))
    max_value = max(list(map(abs, list_copy)))

    def normalize_(n):
        return n / max_value

    list_copy = list(map(normalize_, list_copy))

    return list_copy

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            num_cords = len(handLms.landmark)
            landmarks = ['class']
            for i in range(1, num_cords + 1):
                landmarks += ['x{}'.format(i), 'y{}'.format(i)]

            # data = handLms.landmark
            # data_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in data]).flatten())
            data = cal_landmarks(imgRGB, handLms)
            normalized_data = normalizeData(data)

            data_row = normalized_data
            X = pd.DataFrame([data_row])
            body_language_class = model.predict(X)[0]

            cv2.rectangle(img, (0, 0), (250, 60), (245, 117, 16), -1)
            cv2.putText(img, body_language_class, (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # for rf and lr
            '''body_language_prob = model.predict_proba(X)[0]
            maxProb = round(body_language_prob[np.argmax(body_language_prob)],2)
            cv2.rectangle(img, (0,0), (250,60), (245,117,16), -1)

            if maxProb > 0.6:
                cv2.putText(img, body_language_class, (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                #cv2.putText(img, str(maxProb), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)'''

        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS,
                              landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                              connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

    cv2.imshow("Sign Language Translator", img)
    k = cv2.waitKey(1) & 0xFF

    if(k==27):
        break

cap.release()
cv2.destroyAllWindows()