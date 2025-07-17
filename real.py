# Real-Time Virtual Fitting Room with OpenCV and MediaPipe
# Features: Webcam input, pose estimation, outfit overlay

import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load outfit image (PNG with transparency)
outfit = cv2.imread('D:/PROTOSEM HACKATHON/shirt.png', cv2.IMREAD_UNCHANGED) # Should be RGBA

# Resize outfit to fit body dimensions based on landmarks
def overlay_outfit(frame, outfit, left_shoulder, right_shoulder, left_hip, right_hip):
    # Calculate dimensions and position
    width = int(np.linalg.norm(np.array(right_shoulder) - np.array(left_shoulder)) * 1.5)
    height = int(np.linalg.norm(np.array(right_hip) - np.array(left_shoulder)) * 1.2)
    center_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
    center_y = int((left_shoulder[1] + right_hip[1]) / 2)

    resized_outfit = cv2.resize(outfit, (width, height))
    y1 = max(0, center_y - height // 2)
    x1 = max(0, center_x - width // 2)
    y2 = y1 + height
    x2 = x1 + width

    # Ensure outfit fits within the frame
    if y2 > frame.shape[0] or x2 > frame.shape[1]:
        return frame

    alpha_s = resized_outfit[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(3):
        frame[y1:y2, x1:x2, c] = (alpha_s * resized_outfit[:, :, c] +
                                  alpha_l * frame[y1:y2, x1:x2, c])
    return frame

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        # Extract coordinates for overlay
        l_sh = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        r_sh = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
        l_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                 int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))
        r_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                 int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))

        # Overlay outfit
        frame = overlay_outfit(frame, outfit, l_sh, r_sh, l_hip, r_hip)

        # Optional: draw pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Virtual Fitting Room', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
