import cv2
import numpy as np
import random
from Detection.Get_Var import detect_body

black_hole_x = None
black_hole_y = None
black_hole_radius = 50
particles = []
absorbed_particles = []  # เก็บอนุภาคที่ถูกดูด

# เริ่มต้นอนุภาค
def spawn_new_particles(num_particles=1000):
    """สร้างอนุภาคใหม่เมื่อชูแขนขึ้น"""
    global particles
    h, w = 1080, 1920  # ขนาดจอโปรเจคเตอร์

    for _ in range(num_particles):
        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top":
            x, y = random.randint(0, w), 0
        elif side == "bottom":
            x, y = random.randint(0, w), h
        elif side == "left":
            x, y = 0, random.randint(0, h)
        else:
            x, y = w, random.randint(0, h)

        distance = np.sqrt((x - black_hole_x) ** 2 + (y - black_hole_y) ** 2)
        theta = np.arctan2(y - black_hole_y, x - black_hole_x)

        speed = random.uniform(1, 3)
        particles.append({
            "x": x,
            "y": y,
            "vx": 0,  # ✅ กำหนดค่าเริ่มต้น
            "vy": 0,  # ✅ กำหนดค่าเริ่มต้น
            "speed": speed,
            "theta": theta,
            "distance": distance,
            "opacity": 255  # ✅ กำหนดค่าเริ่มต้นให้อนุภาคมีความโปร่งใสเต็มที่
        })


def update_black_hole_position(frame, hands_up):
    global black_hole_y
    h, w = frame.shape[:2]

    max_black_hole_y = h * 0.2  # ✅ จำกัดให้ขึ้นไปสูงสุด 80% ของความสูงจอ

    if hands_up:
        black_hole_y = max(max_black_hole_y, black_hole_y - 10)  # ✅ ป้องกันการขึ้นไปเกิน 80%


def update_particles():
    global particles, absorbed_particles
    new_particles = []
    drag = 0.95  # ✅ ลดความเร็วของอนุภาคให้นุ่มนวล
    gravity_strength = 0.5  # ✅ ควบคุมแรงดูดของหลุมดำ
    rotation_force = 0.08  # ✅ แรงเหวี่ยงรอบหลุมดำ

    for p in particles:
        dx = black_hole_x - p["x"]
        dy = black_hole_y - p["y"]
        distance = max(np.sqrt(dx**2 + dy**2), 1)

        # ✅ ตรวจสอบว่า 'vx' และ 'vy' มีอยู่หรือไม่ ถ้าไม่มี ให้กำหนดเป็น 0
        if "vx" not in p:
            p["vx"] = 0
        if "vy" not in p:
            p["vy"] = 0

        if p.get("released", False):  # ✅ อนุภาคที่ถูกปล่อยออกมา → ไม่ถูกดูด
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vx"] *= drag  # ✅ ลดความเร็วลงทุกเฟรม
            p["vy"] *= drag  
        else:
            # ✅ หมุนเข้าไปหาหลุมดำแทนที่จะหมุนออก
            force = gravity_strength / distance  
            dx, dy = dy, -dx  # ✅ หมุนทิศทางเพื่อให้อนุภาคโคจรเข้าไปแทนที่จะออก

            p["vx"] += dx * force * rotation_force
            p["vy"] += dy * force * rotation_force

            # ✅ ค่อยๆ ลดระยะทางเข้าใกล้หลุมดำ
            p["vx"] += gravity_strength * (black_hole_x - p["x"]) / distance
            p["vy"] += gravity_strength * (black_hole_y - p["y"]) / distance

            p["x"] += p["vx"]
            p["y"] += p["vy"]

            p["vx"] *= drag  # ✅ ลดแรงเหวี่ยงเพื่อให้ค่อยๆ เคลื่อนเข้าไป
            p["vy"] *= drag  

        # ✅ ถ้าเข้าใกล้หลุมดำมากเกินไป → ดูดหายไป
        if not p.get("released", False) and distance <= black_hole_radius * 1.2:
            absorbed_particles.append(p)
        else:
            new_particles.append(p)  # ✅ เพิ่มเฉพาะอนุภาคที่ยังเคลื่อนที่อยู่

    particles = new_particles  # ✅ ไม่เพิ่ม `absorbed_particles` กลับเข้ามา

# วาดเอฟเฟกต์หลุมดำ
def draw_black_hole(frame):
    cv2.circle(frame, (int(black_hole_x), int(black_hole_y)), black_hole_radius, (0, 0, 0), -1)
    
    for p in particles:
        opacity = p.get("opacity", 255)  # ✅ ใช้ค่าเริ่มต้น 255 ถ้าไม่มี opacity
        if opacity > 0:  # ✅ ไม่วาดอนุภาคที่โปร่งใสหมดแล้ว
            color = (255, 255, 255)  # ✅ OpenCV ไม่รองรับค่าความโปร่งใสในรูปแบบ RGBA
            cv2.circle(frame, (int(p["x"]), int(p["y"])), 2, color, -1)



# ปล่อยอนุภาคที่ถูกดูดไว้เมื่อเอามือลง
def release_particles():
    global particles, absorbed_particles
    h, w = 1080, 1920  # ✅ ขนาดจอโปรเจคเตอร์

    if absorbed_particles:
        new_released_particles = []  # ✅ สร้างลิสต์ใหม่เพื่อปล่อยอนุภาคออก

        for p in absorbed_particles:
            angle = random.uniform(0, 2 * np.pi)  # ✅ กำหนดทิศทางแบบสุ่ม
            speed = random.uniform(10, 20)  # ✅ เพิ่มความเร็วให้อนุภาคพุ่งออกเร็วขึ้น
            p["x"] = black_hole_x + black_hole_radius * np.cos(angle)
            p["y"] = black_hole_y + black_hole_radius * np.sin(angle)
            p["vx"] = np.cos(angle) * speed
            p["vy"] = np.sin(angle) * speed
            p["speed"] = speed
            p["released"] = True  # ✅ ทำเครื่องหมายว่าอนุภาคถูกปล่อยออกมาแล้ว
            p["opacity"] = 255  # ✅ กำหนดค่าความโปร่งใสเริ่มต้น
            new_released_particles.append(p)

        particles.extend(new_released_particles)  # ✅ เพิ่มเฉพาะอนุภาคที่ถูกปล่อยออกมาใหม่
        absorbed_particles = []  # ✅ ล้างรายการอนุภาคที่ถูกดูดเข้าไป

    # ✅ อัปเดตสถานะของอนุภาคที่ถูกปล่อยออกมา
    for p in particles:
        if p.get("released", False):  # ✅ เฉพาะอนุภาคที่ถูกปล่อยออกมา
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["opacity"] -= 10  # ✅ ทำให้ค่อยๆ จางหายไป

    # ✅ ลบอนุภาคที่ออกนอกจอหรือจางหายไปหมดแล้ว
    particles[:] = [p for p in particles if (0 <= p["x"] <= w and 0 <= p["y"] <= h) and p["opacity"] > 0]




def create_interstellar_black_hole(frame, hands_up):
    global black_hole_x, black_hole_y, can_spawn_particles
    h, w = frame.shape[:2]

    if black_hole_x is None or black_hole_y is None:
        black_hole_x = w // 2
        black_hole_y = h // 2

    update_black_hole_position(frame, hands_up)
    update_particles()

    body_box = detect_body(frame)  # ✅ ตรวจจับร่างกาย

    if hands_up:
        can_spawn_particles = True  # ✅ อนุญาตให้สร้างอนุภาคใหม่
    else:
        can_spawn_particles = False  # ✅ หยุดสร้างอนุภาค แต่ยังคงอนุภาคที่มีอยู่

    if can_spawn_particles:
        spawn_new_particles()  # ✅ สร้างอนุภาคเมื่อชูแขนขึ้น

    if body_box is None:  # ✅ ถ้าไม่พบร่างกายในเฟรม
        release_particles()  # ✅ ปล่อยอนุภาคที่ถูกดูดออกมา

    draw_black_hole(frame)
    return frame




