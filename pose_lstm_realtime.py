import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tensorflow as tf
import threading
import h5py
import json

cap = cv2.VideoCapture(0)  # Change this index if necessary
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

custom_objects = {
    'Orthogonal': tf.keras.initializers.Orthogonal
}

# Load model
with h5py.File("lstm-model.h5", 'r') as f:
    model_config = f.attrs.get('model_config')
    model_config = json.loads(model_config)

    for layer in model_config['config']['layers']:
        if 'time_major' in layer['config']:
            del layer['config']['time_major']

    model_json = json.dumps(model_config)
    model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)

    weights_group = f['model_weights']
    for layer in model.layers:
        layer_name = layer.name
        if layer_name in weights_group:
            weight_names = weights_group[layer_name].attrs['weight_names']
            layer_weights = [weights_group[layer_name][weight_name] for weight_name in weight_names]
            layer.set_weights(layer_weights)

lm_list = []
label = "neutral"
neutral_label = "neutral"

def make_landmark_timestep(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm[:63]  # Ensure only the first 63 features are returned

def draw_landmark_on_image(mpDraw, results, frame):
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for lm in results.pose_landmarks.landmark:
        h, w, _ = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0) if label == neutral_label else (0, 0, 255)
    thickness = 2
    lineType = 2
    cv2.putText(img, str(label),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    result = model.predict(lm_list)
    
    # Update this part to recognize punches
    if result[0][0] > 0.5:  # Adjust threshold based on your model's output
        label = "punch"  # Update label to "punch"
    else:
        label = "neutral"  # Update or add other actions if needed
    return str(label)

i = 0
warm_up_frames = 60

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Retrying...")
        continue

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    print("Processing frame...")  # Debug statement
    i += 1
    if i > warm_up_frames:
        if results.pose_landmarks:
            print("Landmarks detected.")  # Debug statement
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            print(len(lm_list))
            if len(lm_list) == 20:
                result = detect(model, lm_list)  # Call detect directly
                print(f"Detection result: {result}")  # Debug statement
                lm_list = []
            frame = draw_landmark_on_image(mpDraw, results, frame)
        else:
            print("No landmarks detected.")  # Debug statement

    frame = draw_class_on_image(label, frame)
    cv2.imshow("image", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
