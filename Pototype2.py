import cv2
import numpy as np
import random
from Detection.Get_Var import detect_body

black_hole_x = None
black_hole_y = None
black_hole_radius = 50
particles = []
absorbed_particles = []

# สร้างอนุภาคใหม่ให้กระจายเป็นชั้นๆ แบบวงโคจรและมีแสงเรืองรอง

def spawn_new_particles(num_particles=1000):
    global particles
    h, w = 1080, 1920

    for _ in range(num_particles):
        orbit_radius = random.uniform(black_hole_radius * 2, w // 2)
        angle = random.uniform(0, 2 * np.pi)
        x = black_hole_x + orbit_radius * np.cos(angle)
        y = black_hole_y + orbit_radius * np.sin(angle)
        speed = random.uniform(1, 3)
        angular_velocity = random.uniform(0.01, 0.05)
        glow_intensity = random.randint(100, 255)  # ความสว่างของอนุภาค

        particles.append({
            "x": x,
            "y": y,
            "orbit_radius": orbit_radius,
            "angle": angle,
            "angular_velocity": angular_velocity,
            "trail": [],
            "opacity": 255,
            "tail_length": random.randint(20, 50),  # เพิ่มความยาวของหาง
            "glow": glow_intensity  # เพิ่มเอฟเฟกต์เรืองแสง
        })

# อัปเดตการเคลื่อนที่ของอนุภาคให้หมุนรอบหลุมดำ

def update_particles():
    global particles, absorbed_particles
    new_particles = []
    for p in particles:
        p["angle"] += p["angular_velocity"]
        p["x"] = black_hole_x + p["orbit_radius"] * np.cos(p["angle"])
        p["y"] = black_hole_y + p["orbit_radius"] * np.sin(p["angle"])
        
        p["trail"].append((p["x"], p["y"]))
        if len(p["trail"]) > p["tail_length"]:
            p["trail"].pop(0)
        
        new_particles.append(p)
    
    particles = new_particles

# วาดอนุภาคแบบมีแสงเรืองรองและหางยาว

def draw_black_hole(frame):
    cv2.circle(frame, (int(black_hole_x), int(black_hole_y)), black_hole_radius, (0, 0, 0), -1)
    for p in particles:
        for i, (tx, ty) in enumerate(p["trail"]):
            alpha = int(p["glow"] * (i / len(p["trail"])) ** 1.5)  # เอฟเฟกต์จางหาย
            color = (alpha, alpha, alpha)
            cv2.circle(frame, (int(tx), int(ty)), 2, color, -1)
        cv2.circle(frame, (int(p["x"]), int(p["y"])), 4, (p["glow"], p["glow"], p["glow"]), -1)

# ฟังก์ชันหลักสร้างหลุมดำที่อนุภาคหมุนรอบตัว พร้อมเอฟเฟกต์สำหรับห้องมืด

def create_interstellar_black_hole(frame, hands_up):
    global black_hole_x, black_hole_y
    h, w = frame.shape[:2]

    if black_hole_x is None or black_hole_y is None:
        black_hole_x = w // 2
        black_hole_y = h // 2
    
    update_particles()
    
    if hands_up:
        spawn_new_particles(50)
    
    draw_black_hole(frame)
    return frame
