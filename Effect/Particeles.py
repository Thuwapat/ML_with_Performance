import random
import cv2
import numpy as np  
import time
from Detection.Get_Var import get_body_mask
from Projector_Connect import projector_width, projector_height
from Utileize import is_arms_raised

# Screen size
width, height = projector_width, projector_height

cv2.namedWindow("Projector", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Particle storage
num_particles = 0
particles = []
particle_trails = {}  # เก็บประวัติตำแหน่งของอนุภาคเพื่อสร้างหาง
previous_particle_frame = None  # ✅ ใช้ Buffer เก็บอนุภาค

# Glitch storage
glitch_particles = []
body_pixels = None  # Store body pixels for fragmentation
glitch_active = False  # Track if glitch effect is active
glitch_duration = 1  # Duration of glitch effect before dispersion
glitch_start_time = None
dispersion_started = False
cooldown_time = 180  # Cooldown time before effect resets
effect_reset_time = None
particle_start_time = {}
moving_center_y = None  # ✅ จุดศูนย์กลางที่เคลื่อนที่ขึ้น

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

def update_gravity_swirl_particles(body_box, elapsed_time, effect_has_trail):
    global particles, particle_trails

    if body_box is None:
        for particle in particles:
            particle["opacity"] -= 5
            if particle["opacity"] <= 0:
                particle_trails.pop(id(particle), None)
                particles.remove(particle)
        return

    x1, y1, x2, y2 = body_box
    body_center_x = (x1 + x2) // 2
    body_center_y = (y1 + y2) // 2

    for _ in range(10):
        new_particle = {
            "x": body_center_x + random.randint(-50, 50),
            "y": body_center_y + random.randint(-50, 50),
            "vx": random.uniform(-5, 5),
            "vy": random.uniform(-5, 5),
            "opacity": 255,
            "size": random.randint(1, 3),
            "glow_intensity": random.randint(100, 255)  # ✅ เพิ่มแสงเรืองรองให้ทุกอนุภาค
        }
        particles.append(new_particle)
        
        if effect_has_trail:
            particle_trails[id(new_particle)] = []

    for particle in particles:
        particle.setdefault("glow_intensity", random.randint(100, 255))  # ✅ ป้องกัน KeyError
        particle["x"] += particle["vx"] * elapsed_time
        particle["y"] += particle["vy"] * elapsed_time
        particle["vx"] *= (0.97 ** elapsed_time)
        particle["vy"] *= (0.97 ** elapsed_time)

        if effect_has_trail:
            trail = particle_trails.get(id(particle), [])
            trail.append((particle["x"], particle["y"]))
            if len(trail) > 10:
                trail.pop(0)
            particle_trails[id(particle)] = trail


def draw_gravity_swirl_particles(frame):
    global previous_particle_frame

    for particle in particles:
        if particle["opacity"] > 0:
            px, py = scale_particle_position(particle["x"], particle["y"])
            glow = particle["glow_intensity"]
            color = (glow, glow, glow)  # ✅ ทำให้อนุภาคมีแสงเรืองรอง
            cv2.circle(frame, (px, py), particle["size"], color, -1)

        # ✅ วาดเส้นทางของอนุภาค (หาง)
        trail = particle_trails.get(id(particle), [])
        for i in range(1, len(trail)):
            prev_px, prev_py = scale_particle_position(trail[i-1][0], trail[i-1][1])
            curr_px, curr_py = scale_particle_position(trail[i][0], trail[i][1])
            alpha = int(particle["glow_intensity"] * (i / len(trail)) ** 1.5)  # ✅ ทำให้หางค่อยๆ จางลง
            color = (alpha, alpha, alpha)
            cv2.line(frame, (prev_px, prev_py), (curr_px, curr_py), color, 2, lineType=cv2.LINE_AA)

def update_body_energy_particles(body_box, elapsed_time, max_particles=500):
    global particles, particle_trails

    if body_box is None:
        return

    x1, y1, x2, y2 = body_box
    body_center_x = (x1 + x2) // 2
    body_center_y = (y1 + y2) // 2

    # ✅ จำกัดจำนวนอนุภาคสูงสุด
    if len(particles) > max_particles:
        particles[:] = particles[-max_particles:]

    for _ in range(5):  # ✅ ลดจำนวนอนุภาคที่สร้างต่อเฟรม
        new_particle = {
            "x": random.randint(x1, x2),
            "y": random.randint(y1, y2),
            "vx": random.uniform(-2, 2),
            "vy": random.uniform(-2, 2),
            "opacity": 255,
            "size": random.randint(3, 5),
            "trail": [],
            "glow_intensity": random.randint(100, 255)
        }
        new_particle["tail_length"] = random.randint(10, 20)  # ✅ กำหนดค่า tail_length
        particles.append(new_particle)
        particle_trails[id(new_particle)] = []

    # ✅ อัปเดตตำแหน่งของอนุภาค พร้อมลบอนุภาคที่ออกจากจอ
    new_particles = []
    for particle in particles:
        # ✅ ป้องกัน KeyError โดยกำหนดค่าเริ่มต้น
        particle.setdefault("tail_length", random.randint(10, 20))

        dx = body_center_x - particle["x"]
        dy = body_center_y - particle["y"]
        distance = max(np.sqrt(dx**2 + dy**2), 1)

        force = 5 / distance  # ✅ ลดแรงโน้มถ่วง
        dx, dy = -dy, dx  
        particle["vx"] += dx * force * elapsed_time
        particle["vy"] += dy * force * elapsed_time

        particle["x"] += particle["vx"]
        particle["y"] += particle["vy"]
        particle["vx"] *= 0.92
        particle["vy"] *= 0.92

        # ✅ ลบอนุภาคที่ออกจากจอ
        if 0 <= particle["x"] <= projector_width and 0 <= particle["y"] <= projector_height:
            new_particles.append(particle)

        # ✅ จำกัดความยาวของหาง
        trail = particle_trails.get(id(particle), [])
        trail.append((particle["x"], particle["y"]))
        if len(trail) > particle["tail_length"]:
            trail.pop(0)
        particle_trails[id(particle)] = trail

    particles = new_particles  # ✅ อัปเดตเฉพาะอนุภาคที่ยังอยู่ในจอ


####### Glitch Effects ########
def extract_body_pixels(frame, body_box):
    global glitch_particles, body_pixels, glitch_active, glitch_start_time, dispersion_started, effect_reset_time

    if body_box is None:
        return

    # ✅ ดึง Mask ของร่างกาย
    body_mask = get_body_mask(frame)

    if body_mask is None:
        return

    x1, y1, x2, y2 = body_box  # ✅ ใช้ Body_box เป็นขอบเขต

    # ✅ ตัดส่วนที่อยู่นอก Body_box ออกจาก Mask
    cropped_mask = np.zeros_like(body_mask)
    cropped_mask[y1:y2, x1:x2] = body_mask[y1:y2, x1:x2]  # ✅ ใช้ Mask เฉพาะภายใน Body_box

    # ✅ แยกพิกเซลร่างกายที่อยู่ภายใน Body_box เท่านั้น
    body_pixels = cv2.bitwise_and(frame, frame, mask=cropped_mask)

    glitch_particles = []
    glitch_active = True
    glitch_start_time = time.time()
    dispersion_started = False
    effect_reset_time = time.time() + cooldown_time  

    # ✅ สร้างอนุภาคจาก Pixel ที่อยู่ใน Mask ภายใน Body_box เท่านั้น
    for i in range(y1, y2, 5):  # ✅ ลูปเฉพาะภายใน Body_box
        for j in range(x1, x2, 5):
            if cropped_mask[i, j] > 0:  # ✅ ตรวจว่าเป็นส่วนของร่างกาย
                color = (255, 255, 255)  # ✅ อนุภาคเป็นสีขาว
                glitch_particles.append({
                    "x": j, "y": i,
                    "vx": 0, "vy": 0,
                    "opacity": 255,
                    "color": color
                })


def dispersion_effect(body_box, projector_height):
    global glitch_particles, particle_start_time, moving_center_y

    if body_box is None:
        return  # ✅ ถ้าไม่มี Body_box ให้รอ ไม่ต้องทำอะไร

    # ✅ หา Center ของร่างกาย
    x1, y1, x2, y2 = body_box
    body_center_x = (x1 + x2) // 2
    body_center_y = (y1 + y2) // 2

    # ✅ กำหนดให้ moving_center_y เป็นตำแหน่งของ Body_center และค่อยๆ ลอยขึ้น
    if moving_center_y is None or moving_center_y > body_center_y:
        moving_center_y = body_center_y

    target_y = projector_height * 0.2  # ✅ เป้าหมาย = 80% ของจอ (ยิ่งค่า y ต่ำ ยิ่งสูง)

    gravity_strength = 1  # ✅ ค่าความแรงที่ดึงอนุภาคเข้าสู่ Body_center
    moving_speed = 2  # ✅ ความเร็วที่ Center ลอยขึ้น
    explosion_triggered = False  # ✅ ตรวจสอบว่าอนุภาคระเบิดแล้วหรือยัง

    new_particles = []
    for particle in glitch_particles:
        dx = body_center_x - particle["x"]
        dy = moving_center_y - particle["y"]  # ✅ ใช้ moving_center_y แทน Body_box

        distance = max(np.sqrt(dx**2 + dy**2), 1)

        # ✅ ดูดอนุภาคเข้าสู่ศูนย์กลางที่กำลังเคลื่อนที่ขึ้น
        if distance > 10:
            particle["vx"] += (dx / distance) * gravity_strength
            particle["vy"] += (dy / distance) * gravity_strength

        # ✅ ค่อยๆ ลดค่า opacity ให้อนุภาคจางหายไป
        particle["opacity"] = max(0, particle["opacity"] - 0.5)

        # ✅ อัปเดตตำแหน่งอนุภาค
        particle["x"] += particle["vx"]
        particle["y"] += particle["vy"]

        # ✅ เก็บเฉพาะอนุภาคที่ยังมองเห็นอยู่
        if particle["opacity"] > 0:
            new_particles.append(particle)

    # ✅ ค่อยๆ ขยับจุดศูนย์กลางขึ้นไป
    moving_center_y -= moving_speed

    explosion_triggered = True
    # ✅ เมื่อจุดศูนย์กลางลอยถึง 80% ของจอ → ระเบิดอนุภาคออก
    #if moving_center_y <= target_y:
    #    explosion_triggered = True

    if explosion_triggered:
        for particle in new_particles:
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(3, 8)
            particle["vx"] = np.cos(angle) * speed
            particle["vy"] = np.sin(angle) * speed

    glitch_particles[:] = new_particles  # ✅ อัปเดตเฉพาะอนุภาคที่ยังเหลืออยู่

def get_dispersion_status():
    global dispersion_started
    return dispersion_started

def update_dispersion(frame, body_box, body_keypoints):
    global glitch_particles, glitch_active, glitch_start_time, dispersion_started, effect_reset_time

    if body_keypoints and is_arms_raised(*body_keypoints):  # 🔥 ใช้เงื่อนไขกางแขนแทน
        extract_body_pixels(frame, body_box)  # เริ่มเอฟเฟกต์
        glitch_start_time = time.time()
        effect_reset_time = glitch_start_time + cooldown_time  

    if glitch_active:
        glitch_active = False
        dispersion_started = True  

    if dispersion_started:
        dispersion_effect(body_box, projector_height)

    return frame

def draw_dispersion(frame):
    for particle in glitch_particles:
        if "color" in particle and particle["opacity"] > 0:
            px, py = scale_particle_position(particle["x"], particle["y"])
            color = particle["color"]
            cv2.circle(frame, (px, py), 2, color, -1)

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

def scale_particle_position(x, y):
    """ปรับขนาดตำแหน่งอนุภาค"""
    scale_x = projector_width / 640
    scale_y = projector_height / 480
    return int(x * scale_x), int(y * scale_y)

def clear_all_particles():
    """Completely clears all particles for switching effects"""
    global particles, particle_trails, glitch_particles ,dispersion_started
    particles = []
    particle_trails = {}
    glitch_particles = []
    dispersion_started = False