import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from PIL import Image


net = cv2.dnn.readNetFromTensorflow("C:/Python310/Models/graph_opt.pb")

# Define body parts and pose pairs
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

def poseDetector(frame, threshold=0.2):
    """Function to detect pose in an image."""
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    inWidth, inHeight = 368, 368

    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight),
                                       (127.5, 127.5, 127.5), swapRB=True, crop=False))
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
        idFrom, idTo = BODY_PARTS[pair[0]], BODY_PARTS[pair[1]]
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    return frame

# Initialize session state variables
if "page" not in st.session_state:
    st.session_state.page = "home"

if "camera_mode" not in st.session_state:
    st.session_state.camera_mode = None

# Home Page
if st.session_state.page == "home":
    st.title("Welcome to Human Pose Estimation")
    st.text("Click below to start pose estimation.")

    if st.button("Go to Pose Estimation"):
        st.session_state.page = "pose_estimation"
        st.rerun()

# Pose Estimation Page
elif st.session_state.page == "pose_estimation":
    st.title("Human Pose Estimation with OpenCV")
    st.text("Upload an image or video, or use your webcam for pose estimation.")

    

    file_buffer = st.file_uploader("Upload an image or video",
                                   type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Capture Image"):
            st.session_state.camera_mode = "image"

    with col2:
        if st.button("Capture Video"):
            st.session_state.camera_mode = "video"
            
    # Handle Camera Image Capture
    if st.session_state.camera_mode == "image":
        img_file_buffer = st.camera_input("Capture an image")
        if img_file_buffer:
            image = np.array(Image.open(img_file_buffer))
            st.image(image, caption="Captured Image", use_container_width=True)
            processed_image = poseDetector(image)
            st.subheader("Pose Estimation Result")
            st.image(processed_image, caption="Estimated Pose", use_container_width=True)

    # Handle Image Upload
    if file_buffer and file_buffer.type in ["image/jpeg", "image/png", "image/jpg"]:
        image = np.array(Image.open(file_buffer))
        st.image(image, caption="Uploaded Image", use_container_width=True)
        processed_image = poseDetector(image)
        st.subheader("Pose Estimation Result")
        st.image(processed_image, caption="Estimated Pose", use_container_width=True)

    # Handle Video Upload
    
    elif file_buffer is not None:
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


    # Handle Webcam Video Capture
    elif st.session_state.camera_mode == "video":
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        if not cap.isOpened():
            st.error("Error: Could not access webcam. Please check permissions.")
        else:
            stop = st.button("Stop Video Capture")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("No frame captured. Exiting...")
                    break

                processed_frame = poseDetector(frame)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(processed_frame, channels="RGB", use_container_width=True)

                if stop:
                    break

            cap.release()
            cv2.destroyAllWindows()

    # Back to Home Button
    if st.button("Back to Home"):
        st.session_state.page = "home"
        st.session_state.camera_mode = None
        st.rerun()
