import time
import numpy as np
import random
import cv2

fireworks_particles = []  # ✅ เก็บอนุภาคพลุ
explosion_particles = []  # ✅ เก็บอนุภาคที่ลอยอยู่ด้านบน
firework_triggered = False  # ✅ ตรวจสอบว่าพลุถูกยิงแล้วหรือยัง
firework_start_time = None  # ✅ เวลาที่เริ่มต้น
has_launched = False  # ✅ ตรวจสอบว่าพลุถูกยิงแล้วหรือยัง

def firework_effect(projector_width, projector_height):
    global fireworks_particles, explosion_particles, firework_triggered
    global firework_start_time, has_launched, waiting_for_explosion

    launch_x = projector_width // 2  # ✅ กึ่งกลางจอ
    launch_y = projector_height - 50  # ✅ ฐานล่างของจอ
    target_y = int(projector_height * 0.2)  # ✅ จุดที่พลุจะระเบิด (80% ของจอ)

    current_time = time.time()

    # ✅ ยิงพลุขึ้นไปเพียงครั้งเดียว
    if not has_launched:
        has_launched = True  # ✅ บันทึกว่าได้ยิงแล้ว
        fireworks_particles.append({
            "x": launch_x,
            "y": launch_y,
            "vx": random.uniform(-1, 1),
            "vy": -random.uniform(8, 12),  # ✅ พุ่งขึ้นไปด้านบน
            "opacity": 255,
            "size": random.randint(2, 5),
            "glow_intensity": 255,
            "trail": []  # ✅ เพิ่มเส้นหางของอนุภาค
        })

    # ✅ อัปเดตตำแหน่งของพลุ และเพิ่มเส้นหาง (Trail)
    new_fireworks = []
    for particle in fireworks_particles:
        particle["x"] += particle["vx"]
        particle["y"] += particle["vy"]
        particle["opacity"] = max(0, particle["opacity"] - 3)

        # ✅ เพิ่มพิกัดปัจจุบันลงในเส้นหาง
        particle["trail"].append((particle["x"], particle["y"]))
        if len(particle["trail"]) > 50:  # ✅ จำกัดความยาวหาง
            particle["trail"].pop(0)

        # ✅ ถ้าพลุถึงระดับระเบิด → หายไปจากระบบ
        if particle["y"] > target_y:
            new_fireworks.append(particle)

    fireworks_particles = new_fireworks

    # ✅ ตรวจสอบว่าพลุถึง `target_y` หรือยัง
    if not firework_triggered and len(fireworks_particles) == 0:
        firework_triggered = True
        firework_start_time = current_time  # ✅ ตั้งค่าเวลาเริ่มต้นระเบิด
        waiting_for_explosion = True  # ✅ เริ่มนับถอยหลัง 3 วินาที

    # ✅ หลังจากรอ 3 วินาที → อนุภาคจะถูกสร้างขึ้นเรื่อยๆ อย่างต่อเนื่อง
    if firework_triggered and (current_time - firework_start_time >= 3):
        for _ in range(2):  # ✅ ค่อยๆ สร้างอนุภาคใหม่ขึ้นเรื่อยๆ
            angle = random.uniform(0, 2 * np.pi)  # ✅ ทิศทางการกระจายแบบสุ่ม
            speed = random.uniform(0.5, 1.5)  # ✅ ความเร็วแบบสุ่ม
            explosion_particles.append({
                "x": launch_x + random.uniform(-150, 150),
                "y": target_y + random.uniform(-50, 50),
                "vx": np.cos(angle) * speed,
                "vy": np.sin(angle) * speed * 0.3,  # ✅ เคลื่อนที่ช้าๆ
                "opacity": 255,
                "size": random.randint(1, 3),
                "glow_intensity": 255
            })

    # ✅ อัปเดตตำแหน่งของอนุภาคที่ลอยอยู่ด้านบน
    new_explosion = []
    for particle in explosion_particles:
        particle["x"] += particle["vx"]
        particle["y"] += particle["vy"]
        particle["opacity"] = max(0, particle["opacity"] - 1)  # ✅ ค่อยๆ จางหายไป

        if particle["opacity"] > 0:
            new_explosion.append(particle)

    explosion_particles = new_explosion



def draw_firework(frame):
    """ วาดเอฟเฟกต์พลุลงบนจอ Projector """
    for particle in fireworks_particles:
        for i in range(1, len(particle["trail"])):
            alpha = int(255 * (i / len(particle["trail"])) ** 1.5)
            color = (alpha, alpha, alpha)
            pt1 = (int(particle["trail"][i - 1][0]), int(particle["trail"][i - 1][1]))
            pt2 = (int(particle["trail"][i][0]), int(particle["trail"][i][1]))
            cv2.line(frame, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)

    for particle in explosion_particles:
        cv2.circle(frame, (int(particle["x"]), int(particle["y"])), particle["size"], (255, 255, 255), -1)

