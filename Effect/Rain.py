import cv2
import numpy as np
import random
from Detection.Get_Var import detect_umbrella

# ตัวแปรเก็บอนุภาคฝน
rain_particles = []
num_drops = 500  # จำนวนหยดฝน

# ฟังก์ชันเริ่มต้นฝน
def initialize_rain():
    global rain_particles
    rain_particles = []
    for _ in range(num_drops):
        x = random.randint(0, 1920)  # ขนาดจอโปรเจคเตอร์
        y = random.randint(0, 1080)
        length = random.randint(5, 15)
        thickness = 2
        wind = random.randint(-1, 2)
        rain_particles.append({"x": x, "y": y, "length": length, "thickness": thickness, "wind": wind})

# อัปเดตฝน
def update_rain():
    global rain_particles
    for drop in rain_particles:
        drop["y"] += drop["length"]  # ฝนตกลงมา
        drop["x"] += drop["wind"]  # มีการเอียงตามลม
        if drop["y"] > 1080:
            drop["y"] = random.randint(-50, 0)  # รีเซ็ตตำแหน่งด้านบน
            drop["x"] = random.randint(0, 1920)

# วาดฝน
def draw_rain(frame):
    for drop in rain_particles:
        cv2.line(frame, (drop["x"], drop["y"]), (drop["x"], drop["y"] + drop["length"]), (0, 0, 0), drop["thickness"])

# ตรวจจับร่ม และควบคุมฝน
def control_rain(frame):
    umbrellas = detect_umbrella(frame)
    if len(umbrellas) > 0:
        print(f"Detected {len(umbrellas)} umbrellas")  # Debug ตรวจจับร่ม
        update_rain()  # อัปเดตการเคลื่อนที่ของฝน
        draw_rain(frame)
