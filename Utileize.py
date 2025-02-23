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
