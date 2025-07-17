import cv2
import numpy as np
import torch
import mediapipe as mp
from torchvision.transforms import Compose

# ------------------ MiDaS Depth Setup ------------------ #
def load_midas_model(device="cpu"):
    model_type = "DPT_Hybrid"  # Use "DPT_Large" or "MiDaS_small" optionally
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform if "DPT" in model_type else midas_transforms.small_transform
    return midas, transform

def estimate_depth(frame, midas, transform, device="cpu"):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

    depth_map = prediction.squeeze().cpu().numpy()
    return cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

# ------------------ Overlay Warp Logic ------------------ #
def overlay_warp(frame, clothing_img, src_pts, dst_pts):
    if src_pts.shape[0] >= 4 and dst_pts.shape[0] >= 4:
        H, _ = cv2.findHomography(src_pts, dst_pts)
        h, w, _ = frame.shape
        warped_clothing = cv2.warpPerspective(clothing_img, H, (w, h))
        
        # Create mask
        mask = (warped_clothing > 0).astype(np.uint8) * 255
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask_binary = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask_binary)

        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        clothing_fg = cv2.bitwise_and(warped_clothing, warped_clothing, mask=mask_binary)

        combined = cv2.add(frame_bg, clothing_fg)
        return combined
    else:
        print(f"⚠️ Not enough points to compute homography: src={len(src_pts)}, dst={len(dst_pts)}")
        return frame

# ------------------ Main Execution ------------------ #
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    midas, midas_transform = load_midas_model(device)

    mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    mp_pose = mp.solutions.pose.Pose(static_image_mode=False)

    # Load a transparent shirt image (RGBA)
    shirt = cv2.imread("shirt.png", cv2.IMREAD_UNCHANGED)
    if shirt is None:
        print("Error: 'shirt.png' not found in working directory.")
        return

    src_h, src_w = shirt.shape[:2]
    src_pts = np.float32([[0, 0], [src_w, 0], [0, src_h]])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res_pose = mp_pose.process(rgb)
        seg = mp_selfie.process(rgb).segmentation_mask

        # Apply segmentation mask
        _, mask_bin = cv2.threshold(seg, 0.5, 255, cv2.THRESH_BINARY)
        body_mask = cv2.cvtColor(mask_bin.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Estimate depth
        depth = estimate_depth(frame, midas, midas_transform, device)
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

        # Warp shirt
        if res_pose.pose_landmarks:
            lm = res_pose.pose_landmarks.landmark
            h, w = frame.shape[:2]

            pts = []
            for idx in [mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                        mp.solutions.pose.PoseLandmark.LEFT_HIP]:
                p = lm[idx.value]
                pts.append([int(p.x * w), int(p.y * h)])

            dst_pts = np.float32(pts)
            frame = overlay_warp(frame, shirt, src_pts, dst_pts)

        # Combine visuals
        blended = cv2.addWeighted(frame, 0.75, body_mask, 0.25, 0)
        stacked = np.hstack([blended, depth_color])

        cv2.imshow("Virtual Fitting Room - Segmentation + Depth + Warping", stacked)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
