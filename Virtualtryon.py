import cv2
import cvzone
from cvzone.PoseModule import PoseDetector
import os
import tkinter as tk

def get_gender_gui():
    selected_gender = {'gender': None}

    def set_gender(gender):
        selected_gender['gender'] = gender
        window.destroy()

    window = tk.Tk()
    window.title("Select Gender")
    window.geometry("300x100")

    tk.Label(window, text="Choose Gender:", font=("Arial", 14)).pack(pady=10)
    tk.Button(window, text="Male", width=10, command=lambda: set_gender("Male")).pack(side="left", padx=30)
    tk.Button(window, text="Female", width=10, command=lambda: set_gender("Female")).pack(side="right", padx=30)

    window.mainloop()
    return selected_gender['gender'] or "Male"

cap = cv2.VideoCapture(0)
detector = PoseDetector()

category = get_gender_gui()

# Load shirts once
shirt_images = {
    'Male': [cv2.imread(f't-shirts/Male/{img}', cv2.IMREAD_UNCHANGED) for img in os.listdir('t-shirts/Male')],
    'Female': [cv2.imread(f't-shirts/Female/{img}', cv2.IMREAD_UNCHANGED) for img in os.listdir('t-shirts/Female')]
}
shirt_index = 0
total_shirts = len(shirt_images[category])

# Load jeans image once
jeans_image = cv2.imread('t-shirts/Bottoms/jeans.png', cv2.IMREAD_UNCHANGED)
if jeans_image is None:
    print("⚠️ Jeans image not found! Please check 't-shirts/Bottoms/jeans.png' path.")
show_jeans = False  # toggle flag

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)

    img = detector.findPose(img)
    lmList, _ = detector.findPosition(img, bboxWithHands=False)

    if lmList:
        try:
            # Shirt landmarks
            left_shoulder = lmList[11]
            right_shoulder = lmList[12]
            left_hip = lmList[23]
            right_hip = lmList[24]
            neck = lmList[0]

            # Shirt size & position
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            shirt_width = int(shoulder_width * 1.4)
            shoulder_y_avg = (left_shoulder[1] + right_shoulder[1]) // 2
            hip_y_avg = (left_hip[1] + right_hip[1]) // 2
            shirt_height = int((hip_y_avg - shoulder_y_avg) * 1.4)
            mid_x = (left_shoulder[0] + right_shoulder[0]) // 2
            anchor_y = int(shoulder_y_avg + (neck[1] - shoulder_y_avg) * 0.5)

            if shirt_width > 0 and shirt_height > 0:
                shirt = shirt_images[category][shirt_index]
                shirt = cv2.resize(shirt, (shirt_width, shirt_height))
                img = cvzone.overlayPNG(img, shirt, (mid_x - shirt_width // 2, anchor_y))

            # Jeans overlay if toggled and jeans image loaded
            if show_jeans and jeans_image is not None:
                left_knee = lmList[25] if len(lmList) > 25 else None
                right_knee = lmList[26] if len(lmList) > 26 else None

                hip_width = abs(right_hip[0] - left_hip[0])
                jeans_width = int(hip_width * 3.0)  # increased width padding
                jeans_mid_x = (left_hip[0] + right_hip[0]) // 2
                jeans_y = hip_y_avg

                if left_knee and right_knee:
                    knee_y_avg = (left_knee[1] + right_knee[1]) // 2
                    jeans_height = int((knee_y_avg - hip_y_avg) * 2.5)  # increased height padding
                else:
                    jeans_height = int(jeans_width * 1.8)  # fallback height

                if jeans_width > 0 and jeans_height > 0:
                    jeans_resized = cv2.resize(jeans_image, (jeans_width, jeans_height))
                    img = cvzone.overlayPNG(img, jeans_resized, (jeans_mid_x - jeans_width // 2, jeans_y))

        except Exception as e:
            print("⚠️ Overlay error:", e)

    cv2.imshow("Virtual Try-On", img)
    key = cv2.waitKey(1)

    if key == ord('d'):
        shirt_index = (shirt_index + 1) % total_shirts
    elif key == ord('a'):
        shirt_index = (shirt_index - 1) % total_shirts
    elif key == ord('j'):  # toggle jeans visibility
        show_jeans = not show_jeans
        print(f"Jeans {'shown' if show_jeans else 'hidden'}")
    elif key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
