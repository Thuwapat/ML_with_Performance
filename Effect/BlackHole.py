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
    """ ตรวจจับโทรศัพท์ในภาพ """
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

def swirl_effect(frame, center=None, radius=200, strength=3.0):
    """ สร้าง Swirl Transform รอบ ๆ center """
    h, w = frame.shape[:2]
    if center is None:
        center = (w//2, h//2)

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    cx, cy = center

    for y in range(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            r = math.sqrt(dx*dx + dy*dy)
            if r < radius:
                theta = strength * (radius - r) / radius
                angle = math.atan2(dy, dx) + theta
                new_x = cx + r * math.cos(angle)
                new_y = cy + r * math.sin(angle)
            else:
                new_x = x
                new_y = y

            map_x[y, x] = new_x
            map_y[y, x] = new_y

    swirl_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return swirl_frame

def create_black_hole_effect(frame):
    """ สร้าง Swirl + หลุมดำ + เปลี่ยนเป็นขาวดำ """
    h, w, _ = frame.shape

   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    
    swirl_frame = swirl_effect(frame_gray, center=(w//2, h//2), radius=250, strength=3.0)

   
    black_hole_layer = np.zeros_like(swirl_frame, dtype=np.uint8)
    hole_radius = 80
    cv2.circle(black_hole_layer, (w//2, h//2), hole_radius, (0, 0, 0), -1)

    
    num_stars = 200
    for _ in range(num_stars):
        rand_x = np.random.randint(0, w)
        rand_y = np.random.randint(0, h)
        star_color = np.random.randint(150, 255)
        cv2.circle(black_hole_layer, (rand_x, rand_y), 1, (star_color, star_color, star_color), -1)

   
    combined = cv2.addWeighted(swirl_frame, 0.8, black_hole_layer, 0.5, 0)

    return combined

def process_frame(frame):
    """
    ถ้าตรวจจับโทรศัพท์ → เปิด Swirl Effect + หลุมดำ
    ถ้าโทรศัพท์หายไป → ค้างเอฟเฟกต์อีก 10 วินาที ก่อนจะ Fade Out
    """
    global black_hole_active, black_hole_start_time

    phones = detect_phone(frame)
    current_time = time.time()

    
    if len(phones) > 0:
        black_hole_active = True
        black_hole_start_time = current_time

   
    if black_hole_active:
        elapsed_time = current_time - black_hole_start_time
        if elapsed_time < black_hole_duration:
            frame = create_black_hole_effect(frame)
        else:
         
            fade_out_time = 3.0
            fade_ratio = (elapsed_time - black_hole_duration) / fade_out_time
            fade_factor = max(0, 1.0 - fade_ratio)

            frame = cv2.addWeighted(create_black_hole_effect(frame), fade_factor, frame, 1.0 - fade_factor, 0)

            
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
        cv2.imshow("Black Hole Swirl Effect", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
