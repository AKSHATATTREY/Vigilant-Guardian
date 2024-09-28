import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("lstm-hand-grasping.h5")

# Initialize Mediapipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Video capture from webcam
cap = cv2.VideoCapture(0)

# Global variables
lm_list = []
label = "neutral"
neutral_label = "neutral"
warm_up_frames = 60
i = 0
no_of_timesteps = 20

# Function to create timestep data from hand landmarks
def make_landmark_timestep(results):
    c_lm = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                c_lm.append(lm.x)
                c_lm.append(lm.y)
                c_lm.append(lm.z)
    return c_lm

# Function to draw landmarks and bounding box on frame
def draw_landmark_on_image(mp_draw, results, frame):
    for hand_landmarks in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame

def draw_bounding_box_and_label(frame, results, label):
    for hand_landmarks in results.multi_hand_landmarks:
        x_min, y_min = 1, 1
        x_max, y_max = 0, 0
        for lm in hand_landmarks.landmark:
            x_min = min(x_min, lm.x)
            y_min = min(y_min, lm.y)
            x_max = max(x_max, lm.x)
            y_max = max(y_max, lm.y)
        h, w, _ = frame.shape
        x_min = int(x_min * w)
        y_min = int(y_min * h)
        x_max = int(x_max * w)
        y_max = int(y_max * h)
        color = (0, 0, 255) if label != neutral_label else (0, 255, 0)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, f"Status: {label}", (x_min, y_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    return frame

# Detection function using the model
def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    result = model.predict(lm_list)
    if result[0][0] > 0.5:
        label = "neutral"
    elif result[0][1] > 0.5:
        label = "resting"
    elif result[0][2] > 0.5:
        label = "holding"
    elif result[0][3] > 0.5:
        label = "gripping"
    elif result[0][4] > 0.5:
        label = "punching"
    return str(label)

# Setup the display window
cv2.namedWindow("Hand Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Detection", 1200, 1000)

while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    i += 1
    if i > warm_up_frames:
        if results.multi_hand_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            if len(lm_list) == no_of_timesteps:
                detect(model, lm_list)
                lm_list = []
            frame = draw_landmark_on_image(mp_draw, results, frame)
            frame = draw_bounding_box_and_label(frame, results, label)
        cv2.imshow("Hand Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
