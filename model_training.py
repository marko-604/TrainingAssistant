import cv2
import os
import numpy as np
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Paths
VIDEO_DIR = "path_to_videos"  # Change this to your video folder
OUTPUT_DIR = "keypoints_data"  # Where extracted keypoints will be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Function to extract keypoints
def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
        else:
            keypoints = np.zeros(33 * 3)  # 33 keypoints, each with x, y, z

        keypoints_list.append(keypoints)

    cap.release()
    return np.array(keypoints_list)


# Process all videos
for exercise in os.listdir(VIDEO_DIR):
    exercise_path = os.path.join(VIDEO_DIR, exercise)
    if not os.path.isdir(exercise_path):
        continue

    for video in os.listdir(exercise_path):
        if video.endswith(".mp4"):
            video_path = os.path.join(exercise_path, video)
            keypoints = extract_keypoints_from_video(video_path)

            output_file = os.path.join(OUTPUT_DIR, f"{exercise}_{video}.npy")
            np.save(output_file, keypoints)  # Save as NumPy array for fast loading

print("Keypoint extraction complete! Data saved in", OUTPUT_DIR)
