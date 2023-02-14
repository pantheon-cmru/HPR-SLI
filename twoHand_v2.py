import cv2
import mediapipe as mp
import copy
import itertools
import pandas as pd
import pickle
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

with open('Final_test_rf.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)

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
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    word = ''
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image,1)


        if not success:
          print("Ignoring empty camera frame.")
          continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        if results.multi_hand_landmarks:
          for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

            mp_drawing.draw_landmarks(
                 image,
                 hand_landmarks,
                 mp_hands.HAND_CONNECTIONS,
                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

            #print(handedness.classification)
            #print(handedness.classification[0].label)


            rightNormalData, leftNormalData = [],[]
            #print(handedness.classification)

            if handedness.classification[0].label == 'Right':
                 rightData = cal_landmarks(image, hand_landmarks)
                 rightNormalData = normalizeData(rightData)
            else:
                leftData = cal_landmarks(image, hand_landmarks)
                leftNormalData = normalizeData(leftData)

            normalizedData = leftNormalData+rightNormalData

            data_row = normalizedData
            X = pd.DataFrame([data_row])
            body_language_class = model.predict(X)[0]

            brect = calc_bounding_rect(image,hand_landmarks)

            image = cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                         (0, 0, 0), 1)
            image = cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 42),
                         (0, 0, 0), -1)

            info_text = handedness.classification[0].label[0:]
            info_text = info_text +' : '+ body_language_class

            #print(str(body_language_class))
            body_language_prob = model.predict_proba(X)[0]
            maxProb = round(body_language_prob[np.argmax(body_language_prob)], 2)

            if maxProb > 0.80:
                print(maxProb)
                word = str(word) + (str(body_language_class))

            #image = cv2.flip(image,1)
            cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            #print(brect)
            # if handedness == 'right':
            #     right=body_language_class
            # elif handedness == 'left':
            #     left=body_language_class
            # else:
            #     print("error")


            #print(body_language_class)
            # cv2.rectangle(image, (0, 0), (250, 100), (245, 117, 16), -1)
            #
            # cv2.putText(image, 'Left: {}'.format(left), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
            #         cv2.LINE_AA)
            # cv2.putText(image, 'Right: {}'.format(right), (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
            #         cv2.LINE_AA)

            # Flip the image horizontally for a selfie-view display.


        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(1) & 0xFF == 27:
          break
    #newvar=np.unique([word])
    #print(word)
    x = word
    #print(x)
    #print(np.unique(x))
    #print(set(x))

    from collections import OrderedDict
    text = (''.join(OrderedDict.fromkeys(x).keys()))
    '''
    # import torch
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    model_name = 't5_gec_model'
    torch_device = 'cpu'  # 'cuda' if torch.cuda.is_available() else
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(torch_device)


    def correct_grammar(input_text, num_return_sequences):
        batch = tokenizer([input_text], truncation=True, padding='max_length', max_length=64, return_tensors="pt").to(
            torch_device)
        translated = model.generate(**batch, max_length=64, num_beams=10, num_return_sequences=num_return_sequences,
                                    temperature=1.5)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text


    #text = input()
    print(correct_grammar(text, num_return_sequences=1))'''

    '''text1 = 'Me is Rashmi'
    print(correct_grammar(text1, num_return_sequences=1))

    text2 = 'I like play games'
    print(correct_grammar(text2, num_return_sequences=1))'''

cap.release()
