import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tensorflow as tf
import threading
import h5py
import json

# Camera setup
cap = cv2.VideoCapture(0)

# Mediapipe hands setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Load the LSTM model
model = tf.keras.models.load_model('lstm-hand-gripping.h5')

lm_list = []
label = "not grasped"
neutral_label = "not grasped"

# Helper functions
def make_landmark_timestep(results):
    c_lm = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                c_lm.extend([lm.x, lm.y, lm.z])
    return c_lm

def draw_landmark_on_image(mpDraw, results, frame):
    for hand_landmarks in results.multi_hand_landmarks:
        mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
    return frame

def draw_bounding_box_and_label(frame, results, label):
    for hand_landmarks in results.multi_hand_landmarks:
        x_min, y_min, x_max, y_max = 1, 1, 0, 0
        for lm in hand_landmarks.landmark:
            x_min, y_min = min(x_min, lm.x), min(y_min, lm.y)
            x_max, y_max = max(x_max, lm.x), max(y_max, lm.y)
        h, w, _ = frame.shape
        x_min, y_min, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)
        
        color = (0, 0, 255) if label != neutral_label else (0, 255, 0)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, f"Status: {label}", (x_min, y_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    return frame

def detect(model, lm_list):
    global label
    lm_list = np.expand_dims(np.array(lm_list), axis=0)
    result = model.predict(lm_list)
    
    if result[0][0] > 0.5:
        label = "not grasped"
    elif result[0][1] > 0.5:
        label = "grasping"
    elif result[0][2] > 0.5:
        label = "carrying"
    elif result[0][3] > 0.5:
        label = "cupping"
    if label in ["grasping", "carrying", "cupping"]:
        label = "grasped"
    return label

# Main loop
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", 1200, 1000)

i, warm_up_frames = 0, 60

while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    i += 1
    if i > warm_up_frames:
        if results.multi_hand_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            if len(lm_list) == 20:
                threading.Thread(target=detect, args=(model, lm_list)).start()
                lm_list = []
            frame = draw_landmark_on_image(mpDraw, results, frame)
            frame = draw_bounding_box_and_label(frame, results, label)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
