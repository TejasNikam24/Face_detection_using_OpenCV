import cv2  # OpenCV for computer vision
import streamlit as st  # Streamlit for interactive web apps
import numpy as np  # NumPy for numerical computing
from PIL import Image  # PIL for image processing


# Function to detect faces in an uploaded image
def detect_faces_in_image(uploaded_image):
    # Convert uploaded image to NumPy array
    img_array = np.array(Image.open(uploaded_image))

    # Load Haar Cascade face detector
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Convert to grayscale for detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show result in Streamlit (convert BGR → RGB for correct colors)
    st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), use_column_width=True)


# Function to detect faces in live camera stream (Streamlit-friendly)
def detect_faces_from_camera():
    st.write("Starting camera... Press 'Stop Camera' to end.")

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Placeholder for video frames
    frame_placeholder = st.empty()

    # Button to stop camera
    stop_button = st.button("Stop Camera")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Load Haar Cascade
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Detect faces
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display frame in Streamlit
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()


# ================= Streamlit UI =================
st.title("Face Detection App")
st.subheader("Detect faces using your webcam or by uploading an image")

# Webcam button
if st.button("Open Camera"):
    detect_faces_from_camera()

# File uploader
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    detect_faces_in_image(uploaded_image)
