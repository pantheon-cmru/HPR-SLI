import copy
import csv
import itertools
import cv2
import mediapipe as mp
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 960)

def cal_landmarks(image, landmarks):
    img_width, img_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):

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
    class_name = "I LOVE YOU"

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            num_cords = len(handLms.landmark)
            landmarks = ['class']
            for i in range(1, num_cords + 1):
                landmarks += ['x{}'.format(i), 'y{}'.format(i)]

            #print(handLms)
            data = cal_landmarks(img,handLms)
            #print(data)
            normalized_data = normalizeData(data)
            #data_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in data]).flatten())
            data_row = normalized_data
            data_row.insert(0, class_name)
            # data_row.insert(0, class_name)
            #print(data_row)
            #print(normalized_data)

            #print(data_row)
            with open('finalDataset.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(data_row)

            # with open('longData.csv', mode='w', newline='') as f:
            #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #     csv_writer.writerow(landmarks)

        mpDraw.draw_landmarks(img,
                              handLms,
                              mpHands.HAND_CONNECTIONS,
                              landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                              connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

    cv2.imshow("Image", img)
    k = cv2.waitKey(1) & 0xFF

    if k==27:
        break

cap.release()
cv2.destroyAllWindows()