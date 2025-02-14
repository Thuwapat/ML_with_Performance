import numpy as np

def calculate_horizontal_angle(left_x, right_x): # Fucntiuon for calculate angle
    return np.degrees(np.arctan2(0, right_x - left_x))  # Ignores Y-axis

