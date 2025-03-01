import time
import numpy as np
import random
import cv2

# อนุภาคเปลือกพลุ (shell) และอนุภาคประกาย (sparks)
shell_particles = []
spark_particles = []

firework_started = False   # ตรวจสอบว่าพลุถูกยิงแล้วหรือยัง
firework_exploded = False  # ตรวจสอบว่าพลุระเบิดแล้วหรือยัง

def firework_effect(projector_width, projector_height):
    """
    สร้างเอฟเฟกต์พลุที่พุ่งขึ้นด้านบน และระเบิดเป็นวงกลมรอบด้าน (ขาวดำ)
    """
    global shell_particles, spark_particles
    global firework_started, firework_exploded

    # จุดปล่อยพลุจากด้านล่าง
    launch_x = projector_width // 2
    launch_y = projector_height - 50
    # จุดที่อยากให้พลุระเบิด (สูงประมาณ 20% ของจอ)
    target_y = int(projector_height * 0.2)

    # ถ้ายังไม่ได้เริ่ม ให้ยิงพลุ 1 ลูก (shell) 
    if not firework_started:
        firework_started = True
        shell_particles.append({
            "x": launch_x,
            "y": launch_y,
            "vx": random.uniform(-1, 1),
            "vy": -random.uniform(10, 13),
            "opacity": 255,
            "size": 5,
            "trail": []  # เก็บตำแหน่งย้อนหลัง (trail) ของ shell
        })

    # อัปเดตตำแหน่งของ shell
    new_shell = []
    for shell in shell_particles:
        # บันทึกตำแหน่งก่อนอัปเดต เพื่อสร้าง trail
        shell["trail"].append((shell["x"], shell["y"]))
        # จำกัดความยาว trail
        if len(shell["trail"]) > 30:
            shell["trail"].pop(0)

        # อัปเดตตำแหน่ง
        shell["x"] += shell["vx"]
        shell["y"] += shell["vy"]
        # ค่อยๆ ลดความเข้ม
        shell["opacity"] = max(0, shell["opacity"] - 2)

        # ถ้าพลุถึงจุดระเบิดหรือความเข้มเหลือน้อย ให้ระเบิด
        if shell["y"] <= target_y or shell["opacity"] <= 50:
            if not firework_exploded:
                firework_exploded = True
                # สร้างประกายไฟกระจายรอบจุดระเบิด
                spawn_radial_sparks(shell["x"], shell["y"], 200)
            # shell นี้ไม่ต้องเก็บต่อ
        else:
            new_shell.append(shell)
    shell_particles[:] = new_shell

    # อัปเดต spark
    new_sparks = []
    for spark in spark_particles:
        # บันทึกตำแหน่งก่อนอัปเดต
        spark["trail"].append((spark["x"], spark["y"]))
        # จำกัดความยาว trail
        if len(spark["trail"]) > 20:
            spark["trail"].pop(0)

        # อัปเดตตำแหน่ง
        spark["x"] += spark["vx"]
        spark["y"] += spark["vy"]
        # ถ้าต้องการแรงโน้มถ่วงเล็กน้อยให้เพิ่ม spark["vy"] += 0.05 ที่นี่
        spark["opacity"] = max(0, spark["opacity"] - 2)

        if spark["opacity"] > 0:
            new_sparks.append(spark)
    spark_particles[:] = new_sparks

def spawn_radial_sparks(center_x, center_y, num_sparks=200):
    """
    สร้างอนุภาค sparks กระจายเป็นวงกลมรอบจุด (center_x, center_y)
    """
    global spark_particles
    for _ in range(num_sparks):
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(3, 7)  # ความเร็วเริ่มต้นของแต่ละ spark
        spark_particles.append({
            "x": center_x,
            "y": center_y,
            "vx": np.cos(angle) * speed,
            "vy": np.sin(angle) * speed,
            "opacity": 255,
            "size": random.randint(2, 4),
            "trail": []  # เก็บตำแหน่งย้อนหลัง (trail) ของ spark
        })

def draw_firework(frame):
    """
    วาดพลุ (shell) และ sparks ลงบนเฟรม (ขาวดำ)
    พร้อมวาด 'เส้น' หรือ 'trail' ของแต่ละอนุภาค
    """
    # สร้าง overlay สำหรับวาดทั้งหมด เพื่อลดการ addWeighted หลายครั้ง
    overlay = frame.copy()

    # วาด shell + trail
    for shell in shell_particles:
        # วาด trail ของ shell เป็นเส้นสีขาว
        for i in range(1, len(shell["trail"])):
            p1 = shell["trail"][i-1]
            p2 = shell["trail"][i]
            cv2.line(overlay, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 255), 2)

        # วาดจุด shell (เป็นวงกลม)
        alpha_shell = shell["opacity"] / 255.0
        cv2.circle(overlay, (int(shell["x"]), int(shell["y"])), shell["size"], (255,255,255), -1)

    # วาด sparks + trail
    for spark in spark_particles:
        # วาด trail ของ spark
        for i in range(1, len(spark["trail"])):
            p1 = spark["trail"][i-1]
            p2 = spark["trail"][i]
            # ทำเส้นสีขาวบาง ๆ
            cv2.line(overlay, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 255), 1)

        # วาดตัว spark
        alpha_spark = spark["opacity"] / 255.0
        cv2.circle(overlay, (int(spark["x"]), int(spark["y"])), spark["size"], (255,255,255), -1)

    # ผสม overlay กับ frame ทีเดียว
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
