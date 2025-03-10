import cv2
import numpy as np
import random
import time

black_hole_x = None
black_hole_y = None
black_hole_radius = 50
max_radius = 600
growth_speed = 2
particles = []

last_hands_up_time = None  # ✅ ใช้เก็บเวลาล่าสุดที่ยกมือขึ้น
delay_before_expand = 3  # ✅ เวลาหน่วงก่อนเริ่มขยาย 
expansion_start_time = None 
black_hole_growth_triggered = False

# ✅ กำหนดจำนวนอนุภาคสูงสุด
MAX_PARTICLES = 300

# สร้างอนุภาคใหม่ให้กระจายเป็นชั้นๆ แบบวงโคจรและมีแสงเรืองรอง
def spawn_new_particles(num_particles):  # ✅ ลดจำนวนที่เกิดใหม่ต่อรอบ
    global particles
    h, w = 1080, 1920

    # ✅ ตรวจสอบว่าจำนวนอนุภาคปัจจุบันไม่เกิน MAX_PARTICLES
    if len(particles) >= MAX_PARTICLES:
        return  # หยุดเพิ่มอนุภาคถ้ามีมากเกินไป

    for _ in range(num_particles):
        orbit_radius = random.uniform(black_hole_radius * 2, w // 2)
        angle = random.uniform(0, 2 * np.pi)
        x = black_hole_x + orbit_radius * np.cos(angle)
        y = black_hole_y + orbit_radius * np.sin(angle)
        speed = random.uniform(1, 3)
        angular_velocity = random.uniform(0.01, 0.05)
        glow_intensity = random.randint(100, 255)

        particles.append({
            "x": x,
            "y": y,
            "orbit_radius": orbit_radius,
            "angle": angle,
            "angular_velocity": angular_velocity,
            "trail": [],
            "opacity": 255,
            "tail_length": random.randint(20, 50),
            "glow": glow_intensity
        })

# อัปเดตการเคลื่อนที่ของอนุภาคให้หมุนรอบหลุมดำ
def update_particles():
    global particles

    new_particles = []
    for p in particles:
        # ✅ อัปเดตระยะโคจรของอนุภาคให้สัมพันธ์กับขนาดของหลุมดำ
        p["orbit_radius"] = max(black_hole_radius * 2, p["orbit_radius"])

        p["angle"] += p["angular_velocity"]
        p["x"] = black_hole_x + p["orbit_radius"] * np.cos(p["angle"])
        p["y"] = black_hole_y + p["orbit_radius"] * np.sin(p["angle"])
        
        p["trail"].append((p["x"], p["y"]))
        if len(p["trail"]) > p["tail_length"]:
            p["trail"].pop(0)
        
        new_particles.append(p)

    # ✅ ลบอนุภาคเก่าหากมีมากเกินไป
    particles = new_particles[-MAX_PARTICLES:]


# วาดอนุภาคแบบมีแสงเรืองรองและหางยาว
def draw_black_hole(frame):
    if black_hole_radius > 0:
        cv2.circle(frame, (int(black_hole_x), int(black_hole_y)), int(black_hole_radius), (0, 0, 0), -1)  # ✅ ใช้ int()

    for p in particles:
        for i in range(1, len(p["trail"])):
            alpha = int(p["glow"] * (i / len(p["trail"])) ** 1.5)
            color = (alpha, alpha, alpha)
            pt1 = (int(p["trail"][i-1][0]), int(p["trail"][i-1][1]))
            pt2 = (int(p["trail"][i][0]), int(p["trail"][i][1]))
            cv2.line(frame, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(frame, (int(p["x"]), int(p["y"])), 4, (p["glow"], p["glow"], p["glow"]), -1)

# ฟังก์ชันหลักสร้างหลุมดำที่อนุภาคหมุนรอบตัว พร้อมเอฟเฟกต์สำหรับห้องมืด
def create_interstellar_black_hole(frame, hands_up):
    global black_hole_x, black_hole_y, black_hole_radius, particles, last_hands_up_time, black_hole_growth_triggered, expansion_start_time
    from Projector_Connect import projector_width, projector_height  

    if black_hole_x is None or black_hole_y is None:
        black_hole_x = projector_width // 2  # ✅ ตั้งค่าให้อยู่กลางจอ
        black_hole_y = projector_height // 2  

    current_time = time.time()

    if hands_up:
        spawn_new_particles(1)  # ✅ สร้างอนุภาคเฉพาะเมื่อยกมือขึ้น
        last_hands_up_time = current_time
        black_hole_growth_triggered = False  # ✅ รีเซ็ตค่าเมื่อยกมือขึ้น
        expansion_start_time = None  # ✅ รีเซ็ตเวลาเริ่มขยาย
    else:
        # ✅ ถ้าเอามือลง และเวลาผ่านไปเกิน 5 วินาที → เริ่มขยายหลุมดำ
        if last_hands_up_time and (current_time - last_hands_up_time > delay_before_expand):
            if not black_hole_growth_triggered:
                expansion_start_time = current_time  # ✅ บันทึกเวลาเริ่มขยาย
            black_hole_growth_triggered = True  

    # ✅ ถ้าอยู่ในโหมดขยาย → ค่อยๆ เพิ่มขนาดหลุมดำแบบ Smooth
    if black_hole_growth_triggered and expansion_start_time:
        time_elapsed = current_time - expansion_start_time
        dynamic_growth_speed = min(10, 2 + time_elapsed * 0.5)  # ✅ ค่อยๆ เร่งความเร็วการขยาย
        black_hole_radius += dynamic_growth_speed
        black_hole_radius = min(black_hole_radius, max_radius)  # ✅ จำกัดขนาดไม่ให้เกินจอ

    # ✅ ถ้าหลุมดำขยายจนเต็มจอ → ทำให้จอเป็นสีดำ และลบอนุภาคทั้งหมด
    if black_hole_radius >= max_radius:
        frame[:] = 0  # ✅ กลืนทั้งจอเป็นสีดำ
        particles.clear()  # ✅ ลบอนุภาคทั้งหมด
        black_hole_growth_triggered = False
        black_hole_radius = 50  # ✅ รีเซ็ตขนาดหลุมดำ
        expansion_start_time = None  # ✅ รีเซ็ตเวลาเริ่มขยาย
    update_particles()
    draw_black_hole(frame)
    return frame