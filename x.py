import cv2
import numpy as np
import math
import time
from ultralytics import YOLO

model_phone = YOLO("yolo11n.pt").to('cuda')

black_hole_active = False
black_hole_start_time = 0
black_hole_duration = 10

def detect_phone(frame):
    results = model_phone.predict(frame)
    phone_boxes = []
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            for i, box in enumerate(boxes):
                if int(class_ids[i]) == 67:
                    phone_boxes.append(box)
    return phone_boxes

def dynamic_swirl_effect(frame, time_elapsed, center=None, radius=200, strength=3.0):
    """สร้าง Swirl Effect ที่หมุนตามเวลา"""
    h, w = frame.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    # สร้าง mapping arrays
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    cx, cy = center

    # เพิ่มการหมุนตามเวลา
    rotation_speed = 1.0  # ปรับความเร็วการหมุน
    base_angle = time_elapsed * rotation_speed

    for y in range(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            r = math.sqrt(dx * dx + dy * dy)
            
            if r < radius:
                # คำนวณมุมพื้นฐาน
                angle = math.atan2(dy, dx)
                
                # เพิ่มการหมุนที่แปรผันตามระยะทางจากจุดศูนย์กลาง
                theta = strength * (radius - r) / radius + base_angle
                
                # ใส่ความเร็วที่แตกต่างกันตามระยะห่างจากจุดศูนย์กลาง
                spiral_factor = 1 - (r / radius)
                theta *= (1 + spiral_factor)
                
                # คำนวณตำแหน่งใหม่
                new_angle = angle + theta
                new_x = cx + r * math.cos(new_angle)
                new_y = cy + r * math.sin(new_angle)
            else:
                new_x = x
                new_y = y

            map_x[y, x] = new_x
            map_y[y, x] = new_y

    # ใช้ remap เพื่อสร้างภาพใหม่
    swirl_frame = cv2.remap(frame, map_x, map_y, 
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, 
                           borderValue=(0, 0, 0))
    return swirl_frame

def process_frame(frame):
    global black_hole_active, black_hole_start_time

    phones = detect_phone(frame)
    current_time = time.time()

    if len(phones) > 0:
        if not black_hole_active:
            black_hole_active = True
            black_hole_start_time = current_time

    if black_hole_active:
        elapsed_time = current_time - black_hole_start_time
        if elapsed_time < black_hole_duration:
            # แปลงภาพเป็นขาวดำ
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # ใช้ dynamic swirl effect
            frame = dynamic_swirl_effect(frame_gray, elapsed_time)
        else:
            
            fade_out_time = 3.0
            fade_ratio = (elapsed_time - black_hole_duration) / fade_out_time
            fade_factor = max(0, 1.0 - fade_ratio)

            effect_frame = dynamic_swirl_effect(frame, elapsed_time)
            frame = cv2.addWeighted(effect_frame, fade_factor,
                                  frame, 1.0 - fade_factor, 0)

            if fade_factor <= 0:
                black_hole_active = False

    return frame

def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)
        cv2.imshow("Dynamic Swirl Effect", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()