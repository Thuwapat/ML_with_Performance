import random
import cv2
import numpy as np  
import time

# Screen size
width, height = 640, 480

# Cooldown Timer
last_burst_time = 0  # Track the last burst event
cooldown_duration = 10  # Cooldown in seconds

# Particle storage
num_particles = 5000
particles = []


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
    global particles
    particles = []

def create_particles_at_hand(hand_center):
    global particles
    if hand_center is None:
        return
    
    hand_x, hand_y = hand_center
    for _ in range(30):  # เพิ่มจำนวนอนุภาคที่เกิดขึ้นเมื่อกำมือ
        particles.append({
            "x": hand_x + random.randint(-15, 15),  # ขยายพื้นที่การกระจายตัว
            "y": hand_y + random.randint(-15, 15),
            "vx": random.uniform(-5, 5),  # เพิ่มความเร็วของอนุภาค
            "vy": random.uniform(-5, 5),
            "opacity": 255,
        })

def update_gravity_swirl_particles(hand_center, hand_open, handful, elapsed_time):
    global particles
    if hand_center is None:
        for particle in particles:
            particle["opacity"] -= 5  # ลดความโปร่งใสของอนุภาคเมื่อไม่มีมือ
            if particle["opacity"] <= 0:
                particles.remove(particle)  # ลบอนุภาคที่จางจนมองไม่เห็นออกจากลิสต์
        return

    hand_x, hand_y = hand_center
    min_force = 10  # เพิ่มแรงดึงดูดของอนุภาค

    if handful:
        create_particles_at_hand(hand_center)  # สร้างอนุภาคที่มือเมื่อกำหมัด

    for particle in particles:
        dx = particle["x"] - hand_x
        dy = particle["y"] - hand_y
        distance = max(np.sqrt(dx**2 + dy**2), 1)
        
        dx /= distance
        dy /= distance
        
        if hand_open:
            force = max(min_force, 5 / distance)  # ลดแรงผลักออกเพื่อให้หมุนช้าลง
            dx, dy = -dy * 0.5, dx * 0.5  # ลดความเร็วของการหมุนออก
            particle["vx"] += dx * force * elapsed_time
            particle["vy"] += dy * force * elapsed_time
        
        particle["x"] += particle["vx"]
        particle["y"] += particle["vy"]
        
        particle["vx"] *= 0.97  # ลด damping เพื่อให้อนุภาคยังคงเคลื่อนที่เร็วขึ้น
        particle["vy"] *= 0.97

def draw_gravity_swirl_particles(frame):
    for particle in particles:
        size = particle.get("size", 3)  # ถ้าไม่มีค่า "size" ให้กำหนดค่าเริ่มต้นเป็น 5
        cv2.circle(frame, (int(particle["x"]), int(particle["y"])), size, (255, 255, 255), -1)



####### Glitch Effects ########
def extract_body_pixels(frame, body_box):
    global glitch_particles, body_pixels, glitch_active, glitch_start_time, dispersion_started, effect_reset_time

    x1, y1, x2, y2 = body_box  
    body_pixels = frame[y1:y2, x1:x2].copy()

    glitch_particles = []
    glitch_active = True
    glitch_start_time = time.time()
    dispersion_started = False
    effect_reset_time = time.time() + cooldown_time  # Ensure effect does not reset immediately

    # Create small pixel fragments for glitch effect
    for i in range(0, body_pixels.shape[0], 5):
        for j in range(0, body_pixels.shape[1], 5):
            if i < body_pixels.shape[0] and j < body_pixels.shape[1]:  
                color = tuple(int(c) for c in body_pixels[i, j])
                glitch_particles.append({
                    "x": x1 + j, "y": y1 + i,
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

def dispersion_effect():
    global glitch_particles
    for particle in glitch_particles:
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(2, 6)  # Increased speed range for more dynamic movement

        # Give more randomness to movement
        particle["vx"] = np.cos(angle) * speed + random.uniform(-2, 2)
        particle["vy"] = np.sin(angle) * speed + random.uniform(-2, 2) 

def get_dispersion_status():
    global dispersion_started
    return dispersion_started

def update_glitch(frame, body_box, hands_together):
    global glitch_particles, glitch_active, glitch_start_time, dispersion_started, effect_reset_time

    if hands_together and body_box and not glitch_active:
        extract_body_pixels(frame, body_box)
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
        dispersion_effect()

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
