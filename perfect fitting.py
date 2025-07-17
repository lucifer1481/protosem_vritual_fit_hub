import cv2
import mediapipe as mp
import numpy as np

# Load shirt images with transparency (PNG)
shirt_images = {
    "front": cv2.imread("shirt_front.png", cv2.IMREAD_UNCHANGED),
    "back": cv2.imread("shirt_back.png", cv2.IMREAD_UNCHANGED),
    "left": cv2.imread("shirt_left.png", cv2.IMREAD_UNCHANGED),
    "right": cv2.imread("shirt_right.png", cv2.IMREAD_UNCHANGED)
}

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

def overlay_transparent(background, overlay, x, y, overlay_size=None):
    if overlay_size:
        overlay = cv2.resize(overlay, overlay_size)

    h, w = overlay.shape[:2]

    # Ensure the overlay fits in the background
    if x < 0:
        overlay = overlay[:, -x:]
        w += x
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        h += y
        y = 0
    if x + w > background.shape[1]:
        overlay = overlay[:, :background.shape[1] - x]
    if y + h > background.shape[0]:
        overlay = overlay[:background.shape[0] - y, :]

    if overlay.shape[2] < 4:
        return background  # not a valid transparent image

    b, g, r, a = cv2.split(overlay)
    mask = a / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = background[y:y+h, x:x+w, c] * (1 - mask) + [b, g, r][c] * mask
    return background

def get_orientation(landmarks):
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    dx = rs.x - ls.x
    dy = rs.y - ls.y
    angle = np.degrees(np.arctan2(dy, dx))

    if -15 <= angle <= 15:
        return "front"
    elif angle > 15:
        return "left"
    elif angle < -15:
        return "right"
    else:
        return "back"

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark
        h, w = frame.shape[:2]

        # Extract key body landmarks
        l_shoulder = np.array([int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),
                               int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h)])
        r_shoulder = np.array([int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
                               int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)])
        l_hip = np.array([int(lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * w),
                          int(lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * h)])
        r_hip = np.array([int(lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w),
                          int(lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h)])

        # Neck = midpoint between shoulders
        neck = ((l_shoulder + r_shoulder) // 2).astype(int)
        # Mid-hip
        mid_hip = ((l_hip + r_hip) // 2).astype(int)

        # Calculate shirt size
        shirt_width = int(np.linalg.norm(r_shoulder - l_shoulder) * 1.25)
        shirt_height = int(np.linalg.norm(mid_hip - neck) * 1.35)

        # Choose correct shirt image based on body orientation
        orientation = get_orientation(lm)
        shirt = shirt_images.get(orientation)

        if shirt is not None:
            top_left_x = neck[0] - shirt_width // 2
            top_left_y = neck[1] - int(0.2 * shirt_height)
            frame = overlay_transparent(frame, shirt, top_left_x, top_left_y, (shirt_width, shirt_height))

        # Optional: Draw landmarks
        # mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Virtual Fitting Room", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
