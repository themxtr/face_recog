import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Paths
database_path = "C:/Users/FRIDAY/Documents/Notes/random pyhton files/jupyterfiles/path/database"
detected_faces_path = "C:/Users/FRIDAY/Documents/Notes/random pyhton files/jupyterfiles/path/detected_faces"
os.makedirs(detected_faces_path, exist_ok=True)
csv_file = os.path.join(detected_faces_path, "detections.csv")

# Load known faces
@st.cache_resource
def load_known_faces():
    known_encodings, known_names = [], []
    for filename in os.listdir(database_path):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(database_path, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_encodings.append(encoding[0])
                known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names

known_face_encodings, known_face_names = load_known_faces()
detected_faces = {}

# Load or initialize attendance
def load_attendance():
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    return pd.DataFrame(columns=["Name", "Attendance"])

df = load_attendance()

def update_attendance(name):
    global df
    if name in df["Name"].values:
        df.loc[df["Name"] == name, "Attendance"] += 1
    else:
        df = pd.concat([df, pd.DataFrame([[name, 1]], columns=["Name", "Attendance"])], ignore_index=True)
    df.to_csv(csv_file, index=False)

# Streamlit UI
st.title("Face Recognition Attendance System")

start_button = st.button("Start Camera")
stop_button = st.button("Stop Camera")

# Initialize session state
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

frame_holder = st.empty()

# Start Camera
if start_button:
    st.session_state.camera_running = True

if stop_button:
    st.session_state.camera_running = False

# Camera Processing Loop (Runs Inside the Main Thread)
if st.session_state.camera_running:
    video_capture = cv2.VideoCapture(0)

    while st.session_state.camera_running:
        ret, frame = video_capture.read()
        if not ret:
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if face_distances.size else None
            name = "Unknown"

            if best_match_index is not None and face_distances[best_match_index] < 0.5:
                name = known_face_names[best_match_index]
                if name not in detected_faces or (datetime.now() - detected_faces[name]).seconds > 10:
                    detected_faces[name] = datetime.now()
                    update_attendance(name)

            top, right, bottom, left = [v * 4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        frame_holder.image(frame, channels="BGR")  # Update UI

    video_capture.release()
    cv2.destroyAllWindows()

st.write("### Attendance Record")
st.dataframe(df)
