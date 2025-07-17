import cv2
import mediapipe as mp
import numpy as np

# Load shirt images
shirt_images = {
    "front": cv2.imread("shirt_front.png", cv2.IMREAD_UNCHANGED),
    "back": cv2.imread("shirt_back.png", cv2.IMREAD_UNCHANGED),
    "left": cv2.imread("shirt_left.png", cv2.IMREAD_UNCHANGED),
    "right": cv2.imread("shirt_right.png", cv2.IMREAD_UNCHANGED)
}

# Initialize pose detector
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Blend transparent PNG over frame
def overlay_transparent(background, overlay, x, y, overlay_size=None):
    if overlay_size:
        overlay = cv2.resize(overlay, overlay_size)

    h, w = overlay.shape[0], overlay.shape[1]
    if x + w > background.shape[1] or y + h > background.shape[0] or x < 0 or y < 0:
        return background

    b, g, r, a = cv2.split(overlay)
    mask = a / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = background[y:y+h, x:x+w, c] * (1 - mask) + [b, g, r][c] * mask
    return background

# Detect body orientation based on shoulder and hip landmarks
def get_orientation(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    shoulder_dx = right_shoulder.x - left_shoulder.x
    shoulder_dy = right_shoulder.y - left_shoulder.y

    angle = np.degrees(np.arctan2(shoulder_dy, shoulder_dx))

    if -15 < angle < 15:
        return "front"
    elif angle <= -15:
        return "right"
    elif angle >= 15:
        return "left"
    else:
        return "back"

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        orientation = get_orientation(landmarks)
        shirt = shirt_images.get(orientation, None)

        # Get coordinates for scaling the shirt
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        frame_h, frame_w, _ = frame.shape

        x1 = int(min(left_shoulder.x, right_shoulder.x) * frame_w)
        x2 = int(max(left_shoulder.x, right_shoulder.x) * frame_w)
        y1 = int(min(left_shoulder.y, right_shoulder.y) * frame_h)
        y2 = int(max(left_hip.y, right_hip.y) * frame_h)

        shirt_width = x2 - x1
        shirt_height = y2 - y1

        if shirt is not None:
            frame = overlay_transparent(frame, shirt, x1, y1, (shirt_width, shirt_height))

        # Optional: draw pose landmarks for debugging
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Virtual Fitting Room", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
