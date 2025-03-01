# Bubble.py
import cv2
import numpy as np
import random
import time

# เก็บข้อมูลฟองอากาศ
bubbles = []
bubble_start_time = None
bubble_duration = 2.0  # ให้ฟองอากาศลอยอยู่ ~2 วินาที
bubble_active = False

def start_bubble_transition():
    """
    เรียกฟังก์ชันนี้เพื่อเริ่มต้นเอฟเฟกต์ Bubble Transition
    """
    global bubble_start_time, bubble_active, bubbles
    bubble_start_time = time.time()
    bubble_active = True
    bubbles = []  # เคลียร์ข้อมูลเก่า
    spawn_bubbles(num_bubbles=40)  # สร้างฟองอากาศตั้งต้น

def spawn_bubbles(num_bubbles=40):
    """
    สร้างฟองอากาศจำนวนหนึ่งอยู่บริเวณด้านล่างจอเพื่อเตรียมลอยขึ้น
    """
    global bubbles
    # สมมุติว่าเราทราบขนาดหน้าจอ (กว้าง x สูง) ไว้ที่ตัวแปร global
    # ถ้าโค้ดหลักคุณใช้ projector_width, projector_height ก็สามารถ import มาใช้ได้เช่นกัน
    width, height = 1920, 1080  # แก้ไขตามจริงหรือ import จาก Projector_Connect
    for _ in range(num_bubbles):
        x = random.randint(0, width)
        y = height + random.randint(10, 200)  # เริ่มต่ำกว่าขอบจอเล็กน้อย
        size = random.randint(15, 40)
        speed = random.uniform(0.5, 2.0)  # ความเร็วลอยขึ้น
        alpha = 255  # ความโปร่งใสเริ่มต้น
        bubbles.append({
            "x": x,
            "y": y,
            "size": size,
            "speed": speed,
            "alpha": alpha
        })

def update_bubbles(frame):
    """
    อัปเดตตำแหน่งของฟองอากาศในแต่ละเฟรม และวาดฟองอากาศลงบน frame
    """
    global bubbles
    height = frame.shape[0]

    for bubble in bubbles:
        # ขยับ bubble ขึ้น
        bubble["y"] -= bubble["speed"]
        # ค่อย ๆ ลด alpha เพื่อให้ค่อย ๆ หาย
        bubble["alpha"] = max(0, bubble["alpha"] - 1)

    # กรองเฉพาะฟองที่ยังไม่หลุดจอและยังไม่จางหมด
    bubbles = [b for b in bubbles if b["y"] + b["size"] > 0 and b["alpha"] > 0]

    # วาดฟองอากาศลงบนภาพ
    for bubble in bubbles:
        # สามารถวาดเป็นวงกลมใส ๆ ขอบขาว หรือจะให้มีไส้สว่างจาง ๆ ก็ได้
        overlay = frame.copy()
        cv2.circle(
            overlay,
            (int(bubble["x"]), int(bubble["y"])),
            bubble["size"],
            (255, 255, 255),
            thickness=2
        )
        # ใช้ alpha blend ถ้าต้องการความโปร่งใส
        alpha_factor = bubble["alpha"] / 255.0
        frame[:] = cv2.addWeighted(overlay, alpha_factor, frame, 1 - alpha_factor, 0)

def is_bubble_transition_finished():
    """
    เช็กว่าเวลาของเอฟเฟกต์บับเบิลหมดหรือยัง
    """
    global bubble_start_time, bubble_duration
    if bubble_start_time is None:
        return True
    return (time.time() - bubble_start_time) > bubble_duration
