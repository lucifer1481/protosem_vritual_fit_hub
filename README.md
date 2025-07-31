Virtual Fitting Room – Real-Time & Image-Based Virtual Try-On

This project was developed during a 3-day hackathon conducted as part of ProtoSem at Forge, KCT Tech Park. It enables users to virtually try on clothes either through a live webcam feed or by uploading a static image. The system uses pose estimation to overlay transparent shirt and jeans images onto the person’s body, creating a virtual fitting experience.

Key Features:

Real-Time Try-On (Webcam):
Uses webcam input to detect human body pose and overlays transparent clothing images in real time. Users can choose gender, switch shirts using keypress, and toggle jeans on/off.

Image-Based Try-On (Upload):
Allows users to upload their own photo along with an outfit image. The system detects pose landmarks and overlays the outfit realistically onto the image.

Technology Stack:

Python

OpenCV (for image processing and webcam access)

MediaPipe (for pose estimation using landmarks)

cvzone (to simplify overlay and landmark access)

Tkinter (for GUI-based gender selection)

OS module (for dynamic image loading)

How It Works:

MediaPipe detects pose landmarks (shoulders, neck, hips, etc.).

Outfit images (shirts/jeans) are resized based on landmark positions.

The transparent images are then placed at appropriate body locations either on a live feed or a static image.

Folder Structure (Example):

t-shirts/
├── Male/
│ ├── shirt1.png
│ └── ...
├── Female/
│ ├── shirt1.png
│ └── ...
└── Bottoms/
  └── jeans.png

Note: All clothing images must be in .png format with transparent backgrounds.

Future Enhancements:

Add support for more clothing types like dresses, skirts, jackets, etc.

Develop a mobile app version with augmented reality (AR) support.

Enhance pose tracking for side poses and multiple users.

Integrate with e-commerce platforms for real-time try-before-you-buy experiences.

Team Members:
Sridha Srinivasaraghavan 
Sanjay S
Saahir Arafath Yasar
Praveen Kumar R
Pooja T

ProtoSem | Forge | KCT Tech Park
