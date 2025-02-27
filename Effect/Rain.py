import cv2
import numpy as np

def add_rain_effect(frame, num_drops=500, min_length=5, max_length=15, thickness=2):
    
    rain_layer = np.zeros_like(frame, dtype=np.uint8)
    h, w, _ = frame.shape

    for _ in range(num_drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        length = np.random.randint(min_length, max_length)
        cv2.line(rain_layer, (x, y), (x, y + length), (255, 255, 255), thickness)

    frame_with_rain = cv2.addWeighted(frame, 0.8, rain_layer, 0.2, 0)
    return frame_with_rain
