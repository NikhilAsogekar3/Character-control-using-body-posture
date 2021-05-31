import cv2
import mediapipe as mp
import numpy as np
from tkinter import *
import random
import time

# Constants
head_size = 100
neck_size = 10
eye_size = 10
sholder_length = 100
body_height = 200
fingure_size = 10

# Canvas settings
tk = Tk()

WIDTH = 800
HEIGHT = 800

canvas = Canvas(tk, width = WIDTH, height = HEIGHT)
tk.title("Nik")
canvas.pack()

# Pose detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    ret, frame = cap.read()

    # Recolor image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)

    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    head = [-300, -300]
    R_hand_mid = [-300, -300]
    R_hand_end = [-300, -300]
    L_hand_mid = [-300, -300]
    L_hand_end = [-300, -300]
    R_leg_mid = [-300, -300]
    R_leg_end = [-300, -300]
    L_leg_mid = [-300, -300]
    L_leg_end = [-300, -300]

    # Extract landmarks
    try:
      landmarks = results.pose_landmarks.landmark

      head = [WIDTH * landmarks[mp_pose.PoseLandmark.NOSE.value].x, HEIGHT * landmarks[mp_pose.PoseLandmark.NOSE.value].y]

      R_hand_mid = [WIDTH * landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, HEIGHT * landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
      R_hand_end = [WIDTH * landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, HEIGHT * landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

      L_hand_mid = [WIDTH * landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, HEIGHT * landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
      L_hand_end = [WIDTH * landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, HEIGHT * landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

      R_leg_mid = [WIDTH * landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, HEIGHT * landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
      R_leg_end = [WIDTH * landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, HEIGHT * landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

      L_leg_mid = [WIDTH * landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, HEIGHT * landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
      L_leg_end = [WIDTH * landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, HEIGHT * landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    except:
      pass

    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )

    head_r = head_size / 2

    head_x1 = head[0] - head_r
    head_y1 = head[1] - head_r
    head_x2 = head_x1 + head_size
    head_y2 = head_y1 + head_size

    head_center_x = head_x1 + head_r
    head_center_y = head_y1 + head_r

    head = canvas.create_oval(head_x1, head_y1, head_x2, head_y2, fill = "pink")
    canvas.create_arc(head_x1, head_y1, head_x2, head_y2 - 3*eye_size, start = 0, extent = 180, fill = "black")

    # Smile
    smile_x1 = head_center_x - 10
    smile_y1 = head_center_y + head_r/2
    smile_x2 = head_center_x + 10
    smile_y2 = head_center_y + head_r/2 + 10

    canvas.create_arc(smile_x1, smile_y1, smile_x2, smile_y2, start =0, extent = -180, fill = "black")

    # Neck
    neck_mid_x = head_center_x
    neck_mid_y = head_center_y + head_r

    neck_x1 = neck_mid_x - 5
    neck_y1 = neck_mid_y
    neck_x2 = neck_mid_x + 5
    neck_y2 = neck_mid_y + neck_size

    neck = canvas.create_rectangle(neck_x1, neck_y1, neck_x2, neck_y2)

    # Eyes
    L_eye_x2 = head_center_x - head_r / 2
    L_eye_y2 = head_center_y
    L_eye_x1 = L_eye_x2 - eye_size
    L_eye_y1 =L_eye_y2 - eye_size

    R_eye_x2 = head_center_x + head_r / 2
    R_eye_y2 = head_center_y
    R_eye_x1 = R_eye_x2 + eye_size
    R_eye_y1 = R_eye_y2 - eye_size

    L_eye = canvas.create_oval(L_eye_x1, L_eye_y1, L_eye_x2, L_eye_y2, fill="blue")
    R_eye = canvas.create_oval(R_eye_x1, R_eye_y1, R_eye_x2, R_eye_y2, fill="blue")

    # Sholder
    sholder_mid_x = head_center_x
    sholder_mid_y = head_center_y + head_r + neck_size
    sholder_x1 = sholder_mid_x - (sholder_length / 2)
    sholder_y1 = sholder_mid_y
    sholder_x2 = sholder_mid_x + (sholder_length / 2)
    sholder_y2 = sholder_mid_y

    # Body
    body_y2 = sholder_y2 + body_height
    body = canvas.create_rectangle(sholder_x1, sholder_y1, sholder_x2, body_y2, fill = "green")

    # Hands
    L_hand_x = sholder_x1
    L_hand_y = sholder_y1
    R_hand_x = sholder_x2
    R_hand_y = sholder_y2

    L_hand_mid_x = L_hand_mid[0]
    L_hand_mid_y = L_hand_mid[1]
    L_hand_end_x = L_hand_end[0]
    L_hand_end_y = L_hand_end[1]

    R_hand_mid_x = R_hand_mid[0]
    R_hand_mid_y = R_hand_mid[1]
    R_hand_end_x = R_hand_end[0]
    R_hand_end_y = R_hand_end[1]

    canvas.create_line(L_hand_x, L_hand_y, L_hand_mid_x, L_hand_mid_y)
    canvas.create_line(L_hand_mid_x, L_hand_mid_y, L_hand_end_x, L_hand_end_y)

    canvas.create_line(R_hand_x, R_hand_y, R_hand_mid_x, R_hand_mid_y)
    canvas.create_line(R_hand_mid_x, R_hand_mid_y, R_hand_end_x, R_hand_end_y)

    # Hand Fingers
    Lh_fingure_x1 = L_hand_end_x - fingure_size
    Lh_fingure_y1 = L_hand_end_y - fingure_size
    Lh_fingure_x2 = L_hand_end_x + fingure_size
    Lh_fingure_y2 = L_hand_end_y + fingure_size

    Rh_fingure_x1 = R_hand_end_x - fingure_size
    Rh_fingure_y1 = R_hand_end_y - fingure_size
    Rh_fingure_x2 = R_hand_end_x + fingure_size
    Rh_fingure_y2 = R_hand_end_y + fingure_size

    canvas.create_oval(Lh_fingure_x1, Lh_fingure_y1, Lh_fingure_x2, Lh_fingure_y2, fill = "black")
    canvas.create_oval(Rh_fingure_x1, Rh_fingure_y1, Rh_fingure_x2, Rh_fingure_y2, fill="black")

    # Legs
    L_leg_x = sholder_x1
    L_leg_y = sholder_y1 + body_height

    R_leg_x = sholder_x2
    R_leg_y = sholder_y2 + body_height

    L_leg_mid_x = L_leg_mid[0]
    L_leg_mid_y = L_leg_mid[1]
    L_leg_end_x = L_leg_end[0]
    L_leg_end_y = L_leg_end[1]

    R_leg_mid_x = R_leg_mid[0]
    R_leg_mid_y = R_leg_mid[1]
    R_leg_end_x = R_leg_end[0]
    R_leg_end_y = R_leg_end[1]

    canvas.create_line(L_leg_x, L_leg_y, L_leg_mid_x, L_leg_mid_y)
    canvas.create_line(L_leg_mid_x, L_leg_mid_y, L_leg_end_x, L_leg_end_y)

    canvas.create_line(R_leg_x, R_leg_y, R_leg_mid_x, R_leg_mid_y)
    canvas.create_line(R_leg_mid_x, R_leg_mid_y, R_leg_end_x, R_leg_end_y)

    # Leg Fingers
    Ll_fingure_x1 = L_leg_end_x - fingure_size
    Ll_fingure_y1 = L_leg_end_y - fingure_size
    Ll_fingure_x2 = L_leg_end_x + fingure_size
    Ll_fingure_y2 = L_leg_end_y + fingure_size

    Rl_fingure_x1 = R_leg_end_x - fingure_size
    Rl_fingure_y1 = R_leg_end_y - fingure_size
    Rl_fingure_x2 = R_leg_end_x + fingure_size
    Rl_fingure_y2 = R_leg_end_y + fingure_size

    canvas.create_oval(Ll_fingure_x1, Ll_fingure_y1, Ll_fingure_x2, Ll_fingure_y2, fill="black")
    canvas.create_oval(Rl_fingure_x1, Rl_fingure_y1, Rl_fingure_x2, Rl_fingure_y2, fill="black")

    tk.update()
    time.sleep(0.01)
    canvas.delete('all')

    cv2.imshow('Nikhil', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

  canvas.mainloop()
  cap.release()
  cv2.destroyAllWindows()




