import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

exercise_mode = "Deadlift"  # Toggle between Deadlift, Squat, and Bench Press


def get_angle(a, b, c):
    ab = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    angle = np.arccos(np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc)))
    return np.degrees(angle)


def check_posture(landmarks, frame, exercise):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

    back_angle = get_angle((left_shoulder.x, left_shoulder.y), (left_hip.x, left_hip.y), (left_knee.x, left_knee.y))
    knee_angle = get_angle((left_hip.x, left_hip.y), (left_knee.x, left_knee.y), (left_ankle.x, left_ankle.y))
    elbow_angle = get_angle((left_shoulder.x, left_shoulder.y), (left_elbow.x, left_elbow.y),
                            (left_wrist.x, left_wrist.y))

    posture_good = True

    if exercise == "Deadlift":
        if not (80 <= back_angle <= 180):
            posture_good = False
            cv2.putText(frame, "Back rounded", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if not (80 <= knee_angle <= 180):
            posture_good = False
            cv2.putText(frame, "Knees too bent or locked", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
    elif exercise == "Squat":
        if knee_angle > 100 or knee_angle < 60:
            posture_good = False
            cv2.putText(frame, "Knee angle off", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    elif exercise == "Bench Press":
        if elbow_angle > 120 or elbow_angle < 60:
            posture_good = False
            cv2.putText(frame, "Elbow angle off", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return posture_good


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3)
            )

            landmarks = results.pose_landmarks.landmark
            posture_good = check_posture(landmarks, frame, exercise_mode)

            if posture_good:
                cv2.putText(frame, f"Perfect {exercise_mode}", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
            else:
                cv2.putText(frame, "Fix Posture", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, f"Exercise Mode: {exercise_mode}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),
                        2, cv2.LINE_AA)

        cv2.imshow('Training Posture Tracker', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            exercise_mode = "Deadlift"
        elif key == ord('2'):
            exercise_mode = "Squat"
        elif key == ord('3'):
            exercise_mode = "Bench Press"

cap.release()
cv2.destroyAllWindows()
