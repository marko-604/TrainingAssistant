import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)


def get_angle(a, b, c):
    # Calculate the angle between three points (a, b, c)
    ab = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    angle = np.arccos(np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc)))
    return np.degrees(angle)


def check_deadlift_posture(landmarks, frame):

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]


    # Back angle: angle between left shoulder, left hip, and left knee
    back_angle = get_angle((left_shoulder.x, left_shoulder.y), (left_hip.x, left_hip.y), (left_knee.x, left_knee.y))

    # Knee angle: angle between knee, ankle, and hip
    knee_angle_left = get_angle((left_hip.x, left_hip.y), (left_knee.x, left_knee.y), (left_ankle.x, left_ankle.y))
    knee_angle_right = get_angle((right_hip.x, right_hip.y), (right_knee.x, right_knee.y),
                                 (right_ankle.x, right_ankle.y))


#---------------------------------------------------------------------------------------------------------------------
#                                                   Deadlift
# ---------------------------------------------------------------------------------------------------------------------

    # check posture thresholds
    posture_good = True
    if not (80 <= back_angle <= 180):
        posture_good = False
        cv2.putText(frame, "Back rounded", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)

    if not (80 <= knee_angle_left <= 180) or not (80 <= knee_angle_right <= 180):  # Loosened knee range
        posture_good = False
        cv2.putText(frame, "Knees too bent or locked", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)

    # check if arms are extended
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    if abs(left_wrist.x - left_shoulder.x) > 0.2 or abs(right_wrist.x - right_shoulder.x) > 0.2:
        posture_good = False
        cv2.putText(frame, "Arms should be extended", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)

    cv2.putText(frame, f"Back Angle: {int(back_angle)}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(frame, f"Knee Angle Left: {int(knee_angle_left)}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Knee Angle Right: {int(knee_angle_right)}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2, cv2.LINE_AA)

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

            # Check posture
            posture_good = check_deadlift_posture(landmarks, frame)


            if posture_good:
                cv2.putText(frame, "Perfect Deadlift", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
            else:
                cv2.putText(frame, "Fix Posture", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Training Posture Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break







cap.release()
cv2.destroyAllWindows()
