#1.IMPORTING MODULES
import tensorflow as tf
import numpy as np
import cv2
import os
import time
import sklearn
import matplotlib.pyplot as plt
import mediapipe as mp

print("All modules imported successfully")

#2.HANDPOINTS AND FACEPOINTS DETECTION WITH MP
# Initialize Mediapipe modules
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
mp_face_mesh = mp.solutions.face_mesh  # For facial connections

def mediapipe_detection(image, model):
    """
    Perform image processing and prediction using the Mediapipe model.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image.flags.writeable = False  # Set image to read-only
    results = model.process(image)  # Process the image with the Mediapipe model
    image.flags.writeable = True  # Make image writeable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR
    return image, results

def draw_landmarks(image, results):
    """
    Draw detected landmarks on the image.
    """
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.face_landmarks, 
            mp_face_mesh.FACEMESH_CONTOURS  # Use FACEMESH_CONTOURS for facial landmarks
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


#3.EXTRACTION OF POINTS FROM CV FEED
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Initialize and test the pipeline
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        _, results = mediapipe_detection(frame, holistic)
        result_test = extract_keypoints(results)
        np.save('0', result_test)  # Save dummy results to file
cap.release()


#4. MAKING FOLDERS

DATA_PATH = os.path.join('MP_Data')  # Path for exported data, numpy arrays
actions = np.array(['A', 'B', 'C'])  # Actions that we try to detect
no_sequences = 30  # Thirty videos worth of data
sequence_length = 30  # Videos are going to be 30 frames in length
start_folder = 0  # Start from folder 0

os.makedirs(DATA_PATH, exist_ok=True)

for action in actions: 
    # Path to action directory
    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)  # Create the action directory if it doesn't exist

    # Get the maximum existing folder number for the action
    try:
        dirmax = max(map(int, filter(str.isdigit, os.listdir(action_path))))
    except ValueError:  # If no directories exist, start from 0
        dirmax = -1

    # Create subdirectories for sequences
    for sequence in range(start_folder, start_folder + no_sequences):
        sequence_path = os.path.join(action_path, str(sequence))
        os.makedirs(sequence_path, exist_ok=True)  # Create the sequence directory
        print(f"Created folder: {sequence_path}")  # Debugging print statement

#5.COLLECTING KEYPOINT VALUES
cap = cv2.VideoCapture(0)
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Loop through actions
    for action in actions:
        for sequence in range(start_folder, start_folder + no_sequences):
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)
            
            for frame_num in range(sequence_length):
                # Read feed
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_landmarks(image, results)

                # Apply wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else:
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                # Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

cap.release()
cv2.destroyAllWindows()
#RUN FROM HERE!!!!!!
#6. PREPROCESSING DATA
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
sequences, labels = [], []
label_map = {label: num for num, label in enumerate(actions)}

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            res = np.load(npy_path)
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Convert to numpy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data preprocessed successfully")
print(f"Train Data Shape: {X_train.shape}")
print(f"Test Data Shape: {X_test.shape}")

#7.MODEL ARCHITECTURE AND TRAINING
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=500, callbacks=[tb_callback])

#7a.SUMMARY OF MODEL
model.summary()

#8.MAKE PREDICTION
res = model.predict(X_test)
actions[np.argmax(res[3])]
actions[np.argmax(y_test[3])]

#9.SAVE WEIGHTS
model.save('action.h5')

#10. EVALUATION USING CONFUSION MATRIX
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
multilabel_confusion_matrix(ytrue, yhat)
accuracy_score(ytrue, yhat)

#11.TESTING REAL TIME
def prob_viz(res, actions, image, colors):
    """Visualize probabilities as a horizontal bar graph overlay on the frame."""
    output_frame = image.copy()
    for i, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + i * 40), (int(prob * 100), 90 + i * 40), colors[i], -1)
        cv2.putText(output_frame, f'{actions[i]}: {prob:.2f}', (5, 80 + i * 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return output_frame
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 16, 117), (16, 245, 245)]

sequence = []  # Store keypoints sequence
sentence = []  # Store recognized actions
predictions = []  # Store prediction indices
threshold = 0.5  # Confidence threshold for predictions

# Initialize Mediapipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Capture video feed
cap = cv2.VideoCapture(0)

# Set up Mediapipe Holistic
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        if not ret:
            break  # If no frame is captured, break out of the loop

        # Make detections
        # Function mediapipe_detection() should preprocess the frame and extract Mediapipe results
        image, results = mediapipe_detection(frame, holistic)
        print(results)  # Debug: Print the results to ensure the model is working

        # Draw landmarks
        draw_landmarks(image, results)  # Function to style the drawn landmarks

        # Extract keypoints for prediction
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep only the last 30 frames for prediction

        # Perform prediction when the sequence is ready
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]  # Predict using the model
            print(actions[np.argmax(res)])  # Debug: Print the action with the highest confidence
            predictions.append(np.argmax(res))

            # Apply visualization and logic for sentence formation
            if np.unique(predictions[-10:])[0] == np.argmax(res):  # Check for consistency in predictions
                if res[np.argmax(res)] > threshold:  # Ensure confidence exceeds the threshold
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:  # Avoid duplicates
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:  # Limit sentence length to 5
                sentence = sentence[-5:]

            # Visualize probabilities (custom function prob_viz to draw confidence bars or graphs)
            image = prob_viz(res, actions, image, colors)

        # Display the sentence on the frame
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)  # Background for text
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the video feed
        cv2.imshow('OpenCV Feed', image)

        # Break loop on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()




