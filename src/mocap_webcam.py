import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import socket
import time


def start_mocap(model_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # h, w, c = frame.shape
        # print(f"Frame shape of the camera: {w} x {h}")

        results = pose.process(rgb_frame)

        frame_array = np.empty((1, 38))

        if results.pose_landmarks:
            # Record pose landmarks x, y coordinates.
            for id, landmark in enumerate(results.pose_landmarks.landmark):
                x, y = np.float32(landmark.x), np.float32(landmark.y)

                # Index can be found here: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index
                if id == 0:  # Nose for head position.
                    # print(f"Nose position: ({x}, {y})")
                    frame_array[0, 0], frame_array[0, 1] = x, y
                elif id == 11:  # Left shoulder
                    frame_array[0, 8], frame_array[0, 9] = x, y
                elif id == 13:  # Left elbow
                    frame_array[0, 10], frame_array[0, 11] = x, y
                elif id == 15:  # Left wrist
                    frame_array[0, 18], frame_array[0, 19] = x, y
                elif id == 21:  # Left thumb
                    frame_array[0, 12], frame_array[0, 13] = x, y
                elif id == 19:  # Left fingers (up)
                    frame_array[0, 4], frame_array[0, 5] = x, y
                elif id == 12:  # Right shoulder
                    frame_array[0, 26], frame_array[0, 27] = x, y
                elif id == 14:  # Right elbow
                    frame_array[0, 28], frame_array[0, 29] = x, y
                elif id == 16:  # Right wrist
                    frame_array[0, 36], frame_array[0, 37] = x, y
                elif id == 22:  # Right thumb
                    frame_array[0, 30], frame_array[0, 31] = x, y
                elif id == 20:  # Right fingers (up)
                    frame_array[0, 22], frame_array[0, 23] = x, y
                elif id == 23:  # Left hip
                    frame_array[0, 2], frame_array[0, 3] = x, y
                elif id == 25:  # Left knee
                    frame_array[0, 14], frame_array[0, 15] = x, y
                elif id == 27:  # Left ankle
                    frame_array[0, 6], frame_array[0, 7] = x, y
                elif id == 31:  # Left toes
                    frame_array[0, 16], frame_array[0, 17] = x, y
                elif id == 24:  # Right hip
                    frame_array[0, 20], frame_array[0, 21] = x, y
                elif id == 26:  # Right knee
                    frame_array[0, 32], frame_array[0, 33] = x, y
                elif id == 28:  # Right ankle
                    frame_array[0, 24], frame_array[0, 25] = x, y
                elif id == 32:  # Right toes
                    frame_array[0, 34], frame_array[0, 35] = x, y

                # Load model
                model = joblib.load(model_path)
                df = pd.DataFrame(
                    frame_array,
                    columns=[
                        "head.x",
                        "head.y",
                        "lfemur.x",
                        "lfemur.y",
                        "lfingers.x",
                        "lfingers.y",
                        "lfoot.x",
                        "lfoot.y",
                        "lhumerus.x",
                        "lhumerus.y",
                        "lradius.x",
                        "lradius.y",
                        "lthumb.x",
                        "lthumb.y",
                        "ltibia.x",
                        "ltibia.y",
                        "ltoes.x",
                        "ltoes.y",
                        "lwrist.x",
                        "lwrist.y",
                        "rfemur.x",
                        "rfemur.y",
                        "rfingers.x",
                        "rfingers.y",
                        "rfoot.x",
                        "rfoot.y",
                        "rhumerus.x",
                        "rhumerus.y",
                        "rradius.x",
                        "rradius.y",
                        "rthumb.x",
                        "rthumb.y",
                        "rtibia.x",
                        "rtibia.y",
                        "rtoes.x",
                        "rtoes.y",
                        "rwrist.x",
                        "rwrist.y",
                    ],
                )
                y_pred = model.predict(df)
                y_pred_list = y_pred.tolist()  # In form of 1D nested list [[]]

                # Send predicted 3D positions to UDP socket.
                # sock.sendto(y_pred_list[0].encode("utf-8"), server_address)

                # time.sleep(1 / fps)  # Set FPS

            # Draw pose landmarks.
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 0, 255), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2
                ),
            )

        cv2.imshow("MediaPipe Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_mocap(model_path="trained_model/MLP_model_all.joblib")
    # start_mocap(model_path="trained_model/XGB_model_all.joblib")

    # Setup UDP
    # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # server_address = ("localhost", 12345)
