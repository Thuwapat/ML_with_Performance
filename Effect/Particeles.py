import random
import cv2
import numpy as np  
import time
from Detection.Get_Var import get_body_mask
projector_width = 1920  
projector_height = 1080 

# Screen size
width, height = projector_width, projector_height

cv2.namedWindow("Projector", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Particle storage
num_particles = 5000
particles = []
particle_trails = {}  # เก็บประวัติตำแหน่งของอนุภาคเพื่อสร้างหาง

# Glitch storage
glitch_particles = []
body_pixels = None  # Store body pixels for fragmentation
glitch_active = False  # Track if glitch effect is active
glitch_duration = 1  # Duration of glitch effect before dispersion
glitch_start_time = None
dispersion_started = False
cooldown_time = 180  # Cooldown time before effect resets
effect_reset_time = None

# Initialize particles in random positions
def initialize_particles():
    global particles, particle_trails
    particles = []
    particle_trails = {}

def create_particles_at_hand(hand_positions):
    global particles, particle_trails
    if not hand_positions:
        return
    
    for hand_x, hand_y in hand_positions:
        for _ in range(30):  # เพิ่มจำนวนอนุภาคที่เกิดขึ้นเมื่อกำมือ
            new_particle = {
                "x": hand_x + random.randint(-15, 15),  # ขยายพื้นที่การกระจายตัว
                "y": hand_y + random.randint(-15, 15),
                "vx": random.uniform(-5, 5),  # เพิ่มความเร็วของอนุภาค
                "vy": random.uniform(-5, 5),
                "opacity": 255,
                "size": random.randint(1, 3),  # ลดขนาดของอนุภาคให้เล็กลงแบบสุ่ม
            }
            particles.append(new_particle)
            particle_trails[id(new_particle)] = []  # เริ่มเก็บข้อมูลเส้นทางของอนุภาค

def update_gravity_swirl_particles(left_hand, right_hand, hand_center, hand_open, handful, elapsed_time):
    global particles, particle_trails
    hand_positions = []
    if hand_center is None:
        for particle in particles:
            particle["opacity"] -= 5  # ลดความโปร่งใสของอนุภาคเมื่อไม่มีมือ
            if particle["opacity"] <= 0:
                particle_trails.pop(id(particle), None)  # ลบเส้นทางของอนุภาคที่หายไป
                particles.remove(particle)  # ลบอนุภาคที่จางจนมองไม่เห็นออกจากลิสต์
        return

    hand_x, hand_y = hand_center
    min_force = 10  # เพิ่มแรงดึงดูดของอนุภาค

    if hand_open:
        if left_hand is not None:
            hand_positions.append(left_hand)
        if right_hand is not None:
            hand_positions.append(right_hand)

        create_particles_at_hand(hand_positions)  # สร้างอนุภาคที่มือเมื่อกำหมัด

    for particle in particles:
        dx = particle["x"] - hand_x
        dy = particle["y"] - hand_y
        distance = max(np.sqrt(dx**2 + dy**2), 1)
        
        dx /= distance
        dy /= distance
        
        if handful:
            force = max(min_force, 5 / distance)  # ลดแรงผลักออกเพื่อให้หมุนช้าลง
            dx, dy = -dy * 0.5, dx * 0.5  # ลดความเร็วของการหมุนออก
            particle["vx"] += dx * force * elapsed_time
            particle["vy"] += dy * force * elapsed_time
        
        particle["x"] += particle["vx"]
        particle["y"] += particle["vy"]
        
        particle["vx"] *= 0.97  # ลด damping เพื่อให้อนุภาคยังคงเคลื่อนที่เร็วขึ้น
        particle["vy"] *= 0.97
        
        # บันทึกเส้นทางของอนุภาคเพื่อสร้างหาง
        trail = particle_trails.get(id(particle), [])
        trail.append((particle["x"], particle["y"]))
        if len(trail) > 10:  # จำกัดความยาวของหางอนุภาค
            trail.pop(0)
        particle_trails[id(particle)] = trail

def draw_gravity_swirl_particles(frame):
    for particle in particles:
        if particle["opacity"] > 0:
            color = (255, 255, 255, int(particle["opacity"]))  # ทำให้อนุภาคจางหายไป
            cv2.circle(frame, (int(particle["x"]), int(particle["y"])), particle["size"], color, -1)
        
        # วาดเส้นทางของอนุภาค (หาง)
        trail = particle_trails.get(id(particle), [])
        for i in range(1, len(trail)):
            alpha = int(255 * (i / len(trail)))  # ทำให้หางค่อยๆ จางลง
            cv2.line(frame, (int(trail[i-1][0]), int(trail[i-1][1])), (int(trail[i][0]), int(trail[i][1])), (255, 255, 255, alpha), 1)

def update_body_energy_particles(body_box, hand_center, hand_open, elapsed_time):
    global particles, particle_trails
    if body_box is None:
        return

    x1, y1, x2, y2 = body_box
    body_center_x = (x1 + x2) // 2
    body_center_y = (y1 + y2) // 2

    for particle in particles:
        dx = body_center_x - particle["x"]
        dy = body_center_y - particle["y"]
        distance = max(np.sqrt(dx**2 + dy**2), 1)

        # ให้อนุภาคเคลื่อนที่โคจรรอบร่างกาย
        force = 8 / distance
        dx, dy = -dy, dx  # หมุน 90 องศาให้เป็นการโคจรรอบ

        particle["vx"] += dx * force * elapsed_time
        particle["vy"] += dy * force * elapsed_time

        # ถ้ามือยกขึ้นเหนือไหล่ → ให้อนุภาคพุ่งออก
        if hand_center is not None and hand_center[1] < y1:
            particle["vx"] += random.uniform(-3, 3)  # เพิ่มแรงกระจาย
            particle["vy"] -= random.uniform(5, 10)

        # อัปเดตตำแหน่ง
        particle["x"] += particle["vx"]
        particle["y"] += particle["vy"]

        particle["vx"] *= 0.95
        particle["vy"] *= 0.95

####### Glitch Effects ########
def extract_body_pixels(frame):
    global glitch_particles, body_pixels, glitch_active, glitch_start_time, dispersion_started, effect_reset_time

    body_mask = get_body_mask(frame)  # ดึง mask ของร่างกาย

    if body_mask is None:
        return

    if body_mask.shape[:2] != frame.shape[:2]:  # ป้องกันปัญหาขนาดไม่ตรงกัน
        body_mask = cv2.resize(body_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    body_pixels = cv2.bitwise_and(frame, frame, mask=body_mask)# ใช้ mask แยกร่างกายออกมา

    glitch_particles = []
    glitch_active = True
    glitch_start_time = time.time()
    dispersion_started = False
    effect_reset_time = time.time() + cooldown_time  

    # สร้างอนุภาคจาก pixel ที่มี mask เท่านั้น
    for i in range(0, body_pixels.shape[0], 5):
        for j in range(0, body_pixels.shape[1], 5):
            if body_mask[i, j] > 0:  # ตรวจว่าเป็นส่วนของร่างกาย
                color = (255, 255, 255)  # อนุภาคเป็นสีขาว
                glitch_particles.append({
                    "x": j, "y": i,
                    "vx": 0, "vy": 0,
                    "opacity": 255,
                    "color": color
                })

def apply_glitch_effect(frame, body_box, glitch_intensity):
    x1, y1, x2, y2 = body_box  

    # Gradually increase glitch intensity over time
    max_shift = int(5 + 25 * glitch_intensity)

    for i in range(y1, y2, 10):
        shift = random.randint(-max_shift, max_shift)
        frame[i:i+5, x1:x2] = np.roll(frame[i:i+5, x1:x2], shift, axis=1)

    for j in range(x1, x2, 10):
        shift = random.randint(-max_shift, max_shift)
        frame[y1:y2, j:j+5] = np.roll(frame[y1:y2, j:j+5], shift, axis=0)

    return frame

def dispersion_effect(body_box):
    global glitch_particles
    if body_box is None:
        return

    # หา Center ของร่างกาย
    x1, y1, x2, y2 = body_box
    body_center_x = (x1 + x2) // 2
    body_center_y = (y1 + y2) // 2

    for particle in glitch_particles:
        # คำนวณทิศทางให้อนุภาคเคลื่อนออกจากร่างกาย
        dx = particle["x"] - body_center_x
        dy = particle["y"] - body_center_y
        distance = max(np.sqrt(dx**2 + dy**2), 1)

        # ปรับให้อนุภาคกระจายออกช้าลง
        speed = 1 + (distance / 50)  # ลดความเร็วของอนุภาค
        particle["vx"] = (dx / distance) * speed
        particle["vy"] = (dy / distance) * speed

        # อัปเดตตำแหน่งอนุภาค
        particle["x"] += particle["vx"]
        particle["y"] += particle["vy"]

    # ลบอนุภาคที่พ้นขอบจอไปไกล
    glitch_particles[:] = [p for p in glitch_particles if -100 <= p["x"] <= width + 100 and -100 <= p["y"] <= height + 100]

def get_dispersion_status():
    global dispersion_started
    return dispersion_started

def update_glitch(frame, body_box, hands_together):
    global glitch_particles, glitch_active, glitch_start_time, dispersion_started, effect_reset_time

    if hands_together and body_box and not glitch_active:
        extract_body_pixels(frame)
        glitch_start_time = time.time()  
        effect_reset_time = glitch_start_time + cooldown_time  

    if glitch_active:
        elapsed_time = time.time() - glitch_start_time if glitch_start_time is not None else 0  
        glitch_intensity = min(1, elapsed_time / glitch_duration)

        if elapsed_time < glitch_duration:
            frame = apply_glitch_effect(frame, body_box, glitch_intensity)
        else:
            glitch_active = False
            dispersion_started = True  

    if dispersion_started:
        dispersion_effect(body_box)

    #  Allow particles to move freely by reducing constraints
    if effect_reset_time is None or time.time() < effect_reset_time:  
        glitch_particles[:] = [p for p in glitch_particles if p["opacity"] > 0]  
        for particle in glitch_particles:
            particle["x"] += particle["vx"] + random.uniform(-1, 1)  # Add slight randomness to movement
            particle["y"] += particle["vy"] + random.uniform(-1, 1)  

            # Reduce damping effect to keep particles moving longer
            particle["vx"] *= 0.95  
            particle["vy"] *= 0.95  

            particle["opacity"] -= 1  

            # Keep particles within screen
            particle["x"] = np.clip(particle["x"], 0, width)
            particle["y"] = np.clip(particle["y"], 0, height)

    return frame

def draw_glitch(frame):
    for particle in glitch_particles:
        if "color" in particle and particle["opacity"] > 0:
            color = particle["color"]
            cv2.circle(frame, (int(particle["x"]), int(particle["y"])), 2, color, -1)

def update_body_orbit_particles(body_box, elapsed_time):
    global particles, particle_trails
    if body_box is None:
        return
    
    x1, y1, x2, y2 = body_box
    body_center_x = (x1 + x2) // 2
    body_center_y = (y1 + y2) // 2

    for particle in particles:
        dx = particle["x"] - body_center_x
        dy = particle["y"] - body_center_y
        distance = max(np.sqrt(dx**2 + dy**2), 1)
        
        force = 8 / distance  # ปรับแรงเพื่อให้อนุภาคโคจรรอบตัว
        angle = np.arctan2(dy, dx) + 0.05  # เพิ่มการหมุน
        
        particle["vx"] = np.cos(angle) * force * elapsed_time
        particle["vy"] = np.sin(angle) * force * elapsed_time
        
        particle["x"] += particle["vx"]
        particle["y"] += particle["vy"]
        
        particle["vx"] *= 0.97  # ลด damping เพื่อให้อนุภาคยังคงเคลื่อนที่
        particle["vy"] *= 0.97