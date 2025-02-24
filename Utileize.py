import numpy as np
from collections import deque

def calculate_horizontal_angle(left_x, right_x): # Fucntiuon for calculate angle
    return np.degrees(np.arctan2(0, right_x - left_x))  # Ignores Y-axis

speed_buffer = deque(maxlen=5)
previous_left_shoulder = None
previous_right_shoulder = None
previous_time = None

def calculate_shoulder_speed(left_shoulder_x, right_shoulder_x, current_time):
    """ คำนวณความเร็วของไหล่ซ้ายและขวา โดยใช้ Moving Average """
    global previous_left_shoulder, previous_right_shoulder, previous_time, speed_buffer

    # ถ้าไม่มีค่าก่อนหน้าให้รีเซ็ตค่า
    if left_shoulder_x is None or right_shoulder_x is None:
        previous_left_shoulder = left_shoulder_x
        previous_right_shoulder = right_shoulder_x
        previous_time = current_time
        return 0  

    if previous_left_shoulder is None or previous_right_shoulder is None or previous_time is None:
        previous_left_shoulder = left_shoulder_x
        previous_right_shoulder = right_shoulder_x
        previous_time = current_time
        return 0  

    time_diff = current_time - previous_time
    if time_diff == 0:
        return 0

    # ✅ คำนวณความเร็วโดยใช้ระยะห่างของไหล่ซ้าย-ขวา
    left_speed = abs(left_shoulder_x - previous_left_shoulder) / time_diff
    right_speed = abs(right_shoulder_x - previous_right_shoulder) / time_diff

    # ✅ เก็บค่าความเร็วใน Buffer (เพื่อทำ Moving Average)
    avg_speed = (left_speed + right_speed) / 2
    speed_buffer.append(avg_speed)

    # ✅ อัปเดตค่าเก่า
    previous_left_shoulder = left_shoulder_x
    previous_right_shoulder = right_shoulder_x
    previous_time = current_time

    # ✅ คืนค่า Moving Average ของค่า Speed
    return np.mean(speed_buffer)

def is_arms_raised(left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, threshold=150):
    """ เช็คว่าทำท่ากางแขนหรือไม่ """

    if None in [left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist]:
        return False  # ถ้าข้อมูลหายไปบางส่วน ให้ return False

    left_shoulder_y, right_shoulder_y = left_shoulder[1], right_shoulder[1]
    left_elbow_y, right_elbow_y = left_elbow[1], right_elbow[1]
    left_wrist_y, right_wrist_y = left_wrist[1], right_wrist[1]

    # ข้อมือสูงกว่าหรือใกล้ระดับไหล่
    wrist_near_shoulder = (
        abs(left_wrist_y - left_shoulder_y) < threshold and abs(right_wrist_y - right_shoulder_y) < threshold
    )

    # ศอกอยู่ระหว่างไหล่และข้อมือ
    elbow_position_valid = (
        left_shoulder_y < left_elbow_y < left_wrist_y and right_shoulder_y < right_elbow_y < right_wrist_y
    )

    return wrist_near_shoulder and elbow_position_valid


