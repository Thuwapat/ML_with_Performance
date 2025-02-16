import cv2
import numpy as np
import math
import time
from ultralytics import YOLO

# โหลดโมเดลตรวจจับโทรศัพท์
model_phone = YOLO("yolo11n.pt").to('cuda')

black_hole_active = False
black_hole_start_time = 0
black_hole_duration = 10  # ระยะเวลาที่เอฟเฟกต์ดำแสดงอยู่

def detect_phone(frame):
    """ตรวจจับโทรศัพท์ในภาพ"""
    results = model_phone.predict(frame)
    phone_boxes = []

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            for i, box in enumerate(boxes):
                if int(class_ids[i]) == 67:  # รหัสสำหรับโทรศัพท์ (ปรับให้ตรงกับโมเดลของคุณ)
                    phone_boxes.append(box)
    return phone_boxes

def swirl_effect(frame, center=None, radius=200, strength=3.0, angle_offset=0):
    """สร้าง Swirl Transform รอบ ๆ center แบบเวกเตอร์ไลซ์"""
    h, w = frame.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    cx, cy = center

    # สร้าง grid ของพิกเซล
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)

    # คำนวณมุมหมุนเฉพาะในพื้นที่ที่อยู่ภายใน radius
    theta = np.zeros_like(r)
    mask = r < radius
    theta[mask] = strength * (radius - r[mask]) / radius

    new_angle = np.arctan2(dy, dx) + theta + angle_offset
    new_x = cx + r * np.cos(new_angle)
    new_y = cy + r * np.sin(new_angle)

    # สำหรับพิกเซลที่อยู่นอก radius ให้ใช้ตำแหน่งเดิม
    new_x[~mask] = x[~mask]
    new_y[~mask] = y[~mask]

    map_x = new_x.astype(np.float32)
    map_y = new_y.astype(np.float32)

    swirl_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return swirl_frame

def create_vignette_mask(shape, strength=0.8):
    """สร้าง Vignette Mask เพื่อให้ภาพมีความรู้สึกเหมือนหลุมดำ"""
    h, w = shape[:2]
    y, x = np.indices((h, w))
    cx, cy = w / 2, h / 2
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_distance = np.sqrt(cx**2 + cy**2)
    mask = 1 - strength * (distance / max_distance)
    mask = np.clip(mask, 0, 1)
    mask = mask[..., np.newaxis]  # ทำให้เป็น 3 มิติสำหรับการคูณกับแต่ละ channel
    return mask

def create_black_hole_effect(frame, time_elapsed):
    """สร้างเอฟเฟกต์หลุมดำ: Swirl + เปลี่ยนเป็นขาวดำ + Vignette"""
    h, w, _ = frame.shape

    # แปลงภาพเป็นขาวดำเพื่อความรู้สึกที่เข้มข้น
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # คำนวณมุมหมุนตามเวลา
    rotation_speed = 1.5  # ปรับความเร็วการหมุนได้ตามต้องการ
    angle_offset = time_elapsed * rotation_speed

    # สร้างเอฟเฟกต์ Swirl
    swirl_frame = swirl_effect(frame_gray, center=(w // 2, h // 2), radius=450, strength=7, angle_offset=angle_offset)

    # เพิ่ม Vignette เพื่อให้ขอบภาพมืดลง คล้ายหลุมดำ
    vignette_mask = create_vignette_mask(frame_gray.shape, strength=0.8)
    black_hole_frame = (swirl_frame * vignette_mask).astype(np.uint8)

    return black_hole_frame

def process_frame(frame):
    """
    ถ้าตรวจจับโทรศัพท์ → เปิดเอฟเฟกต์หลุมดำ (Swirl + Vignette)
    ถ้าโทรศัพท์หายไป → ค้างเอฟเฟกต์อีก 10 วินาที ก่อนจะ Fade Out
    """
    global black_hole_active, black_hole_start_time

    phones = detect_phone(frame)
    current_time = time.time()

    # ถ้ามีการตรวจจับโทรศัพท์และเอฟเฟกต์ยังไม่ active ให้เริ่มเอฟเฟกต์
    if len(phones) > 0:
        if not black_hole_active:
            black_hole_active = True
            black_hole_start_time = current_time

    if black_hole_active:
        elapsed_time = current_time - black_hole_start_time
        if elapsed_time < black_hole_duration:
            frame = create_black_hole_effect(frame, elapsed_time)
        else:
            # ทำ Fade Out ใน 3 วินาที
            fade_out_time = 3.0
            fade_ratio = (elapsed_time - black_hole_duration) / fade_out_time
            fade_factor = max(0, 1.0 - fade_ratio)

            effect_frame = create_black_hole_effect(frame, elapsed_time)
            frame = cv2.addWeighted(effect_frame, fade_factor, frame, 1.0 - fade_factor, 0)

            # เมื่อ fade out เสร็จ ให้ปิดเอฟเฟกต์
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
        cv2.imshow("Black Hole Effect", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
