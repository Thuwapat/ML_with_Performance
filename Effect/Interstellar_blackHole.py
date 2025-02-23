import cv2
import numpy as np

def create_interstellar_black_hole(frame, shoulder_speed, center=None, radius=200, glow_intensity=1.5):
    h, w = frame.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    # ✅ ปรับความโค้งของแสงให้ขึ้นกับความเร็วไหล่
    warp_factor = np.clip(shoulder_speed / 100, 0.2, 1.5)

    glow_layer = np.zeros_like(frame, dtype=np.uint8)
    for i in range(3):
        cv2.circle(glow_layer, center, radius + 10 + i * 5, (255, 255, 255), thickness=2)
    glow_layer = cv2.GaussianBlur(glow_layer, (55, 55), 10) 

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            dx = x - center[0]
            dy = y - center[1]
            r = np.sqrt(dx**2 + dy**2)
            
            if r < radius:
                theta = np.arctan2(dy, dx)
                new_x = center[0] + r * np.cos(theta + warp_factor * np.exp(-r / radius))  
                new_y = center[1] + r * np.sin(theta + warp_factor * np.exp(-r / radius))
            else:
                new_x, new_y = x, y

            map_x[y, x] = new_x
            map_y[y, x] = new_y

    warped_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    black_hole_effect = cv2.addWeighted(warped_frame, 0.8, glow_layer, glow_intensity, 0)
    cv2.circle(black_hole_effect, center, int(radius * 0.7), (0, 0, 0), -1)

    return black_hole_effect

