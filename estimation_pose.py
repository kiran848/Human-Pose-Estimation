import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

# Define body parts and pose pairs
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

# Load OpenPose Model
net = cv2.dnn.readNetFromTensorflow("C:/Python310/Models/graph_opt.pb")
st.title("Human Pose Estimation using OpenCV")
st.text("Upload an image or video, or use your webcam for pose estimation.")

# File Uploader for Image or Video
file_buffer = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# Camera Input
img_file_buffer = st.camera_input("Capture an image")  

# Wait for input processing
time.sleep(2)

def poseDetector(frame, threshold=0.2):
    
    """Function to detect pose in an image."""
    
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    inWidth = 368
    inHeight = 368

    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > threshold else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    return frame

# Handle image input
if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
    st.image(image, caption="Captured Image", use_container_width=True)
    processed_image = poseDetector(image)
    st.subheader("Pose Estimation Result")
    st.image(processed_image, caption="Estimated Pose", use_container_width=True)

# Handle video input
elif file_buffer and file_buffer.type in ["video/mp4", "video/avi", "video/mov"]:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(file_buffer.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()  # Create a placeholder for video frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = poseDetector(frame)

        # Convert frame to RGB for Streamlit display
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # Show frame in Streamlit
        stframe.image(processed_frame, channels="RGB", use_container_width=True)


    cap.release()
    
    cv2.destroyAllWindows()
import os

# Button to launch real-time pose estimation
if "video_mode" not in st.session_state:
    st.session_state.video_mode = False  # False means camera is enabled

# Button to launch real-time pose estimation
if st.button("Start Real-Time Pose Estimation"):
    st.session_state.video_mode = True  # Set to True when button is clicked
    st.success("Launching real-time video pose estimation...")

    # Run `video_estimation.py` in a new terminal using Streamlit
    os.system("start cmd /k python -m streamlit run video_estimation.py")

# Disable camera if video mode is active
if not st.session_state.video_mode:
    file_buffer = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    img_file_buffer = st.camera_input("Capture an image")  
else:
    st.warning("Camera is disabled because real-time pose estimation is running.")

