import random
import cv2
import numpy as np  
import time

# Screen size
width, height = 640, 480

# Cooldown Timer
last_burst_time = 0  # Track the last burst event
cooldown_duration = 10  # Cooldown in seconds

# Snowflake storage
num_flakes = 100
snowflakes = []

# Particle storage
num_particles = 500
particles = []

glitch_particles = []
body_pixels = None  # Store body pixels for fragmentation
glitch_active = False  # Track if glitch effect is active
glitch_duration = 0.5  # Duration of glitch effect before dispersion
glitch_start_time = None

particle_img = cv2.imread("./images/rose.png", cv2.IMREAD_UNCHANGED)  
particle_size = 15  # Adjust size of particle image

# Generate snowflakes
def create_snowflakes():
    global snowflakes
    snowflakes = [{"x": random.randint(0, width), "y": random.randint(0, height), "size": random.randint(2, 6)} for _ in range(num_flakes)]

# Make snow move downward and reset when reaching bottom
def update_snowflakes():
    global snowflakes
    for flake in snowflakes:
        flake["y"] += random.randint(1, 4)  # Snowfall speed
        flake["x"] += random.choice([-1, 0, 1])  # Slight sideways drift
        if flake["y"] > height:
            flake["y"] = 0  # Reset to top
            flake["x"] = random.randint(0, width)

# Draw snowflakes
def draw_snowflakes(frame):
    for flake in snowflakes:
        cv2.circle(frame, (flake["x"], flake["y"]), flake["size"], (255, 255, 255), -1)

####### Particle Effects ########
def create_particles():
    global particles
    for _ in range(num_particles):
        side = random.choice(["top", "bottom", "left", "right"])  # Random border spawn

        if side == "top":
            x, y = random.randint(0, width), 0
        elif side == "bottom":
            x, y = random.randint(0, width), height
        elif side == "left":
            x, y = 0, random.randint(0, height)
        else:
            x, y = width, random.randint(0, height)

        particles.append({"x": x, "y": y, "vx": 0, "vy": 0, "size": random.randint(2, 5), "opacity": 0})  
        
def reset_particles_to_borders():
    global particles
    for particle in particles:
        side = random.choice(["top", "bottom", "left", "right"])

        if side == "top":
            particle["x"], particle["y"] = random.randint(0, width), 0
        elif side == "bottom":
            particle["x"], particle["y"] = random.randint(0, width), height
        elif side == "left":
            particle["x"], particle["y"] = 0, random.randint(0, height)
        else:
            particle["x"], particle["y"] = width, random.randint(0, height)

        particle["vx"], particle["vy"] = 0, 0  
        particle["opacity"] = 0

def heart_shape(index, scale=2):
    t = index * (2 * np.pi / num_particles)
    x = scale * (16 * np.sin(t)**3)
    y = -scale * (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))  # Inverted Y for OpenCV
    return x, y

## If palm is open, particles burst outward and fade away
## If handful is made, particles arrange into a shape
def update_particles(hand_center, hand_open, handful, elapsed_time): 
    global particles, last_burst_time

    current_time = time.time()
    cooldown_active = (current_time - last_burst_time < cooldown_duration)

    if hand_center is None:
        reset_particles_to_borders()
        return  

    hand_x, hand_y = hand_center  

    for i, particle in enumerate(particles):
        if hand_open:
            particle["opacity"] = 255  
            direction_x = particle["x"] - hand_x
            direction_y = particle["y"] - hand_y
            distance = max(np.sqrt(direction_x**2 + direction_y**2), 1)

            direction_x /= distance
            direction_y /= distance

            speed = 6  
            particle["vx"] += direction_x * speed * elapsed_time + random.uniform(-1, 1)
            particle["vy"] += direction_y * speed * elapsed_time + random.uniform(-1, 1)

            particle["opacity"] -= 10  
            if particle["opacity"] <= 0 and not cooldown_active:
                last_burst_time = current_time  
                reset_particles_to_borders()

        if handful:
            target_x, target_y = heart_shape(i, scale=10)
            target_x += hand_x  
            target_y += hand_y

            particle["vx"] += (target_x - particle["x"]) * 0.1 * elapsed_time
            particle["vy"] += (target_y - particle["y"]) * 0.1 * elapsed_time 

        particle["x"] += particle["vx"]
        particle["y"] += particle["vy"]
        particle["vx"] *= 0.9  
        particle["vy"] *= 0.9

        particle["x"] = np.clip(particle["x"], 0, width)
        particle["y"] = np.clip(particle["y"], 0, height)

def overlay_image(background, overlay, x, y, opacity=1.0):
    """
    Overlays an image (overlay) onto a background at position (x, y) with transparency.
    """
    h, w, c = overlay.shape
    x, y = int(x - w / 2), int(y - h / 2)  

    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return background  

    overlay_rgb = overlay[:, :, :3]  # Extract RGB
    if c == 4:
        mask = overlay[:, :, 3] / 255.0 * opacity  # Use alpha channel
    else:
        mask = np.ones((h, w)) * opacity  # If no alpha, use full opacity

    roi = background[y:y + h, x:x + w]

    for c in range(3):  
        roi[:, :, c] = (roi[:, :, c] * (1 - mask) + overlay_rgb[:, :, c] * mask).astype(np.uint8)

    background[y:y + h, x:x + w] = roi  
    return background

def draw_particles(frame):
    """
    Draws floating particles using a .png image.
    """
    global particle_img

    if particle_img is None:
        print("Error: Particle image not found!")
        return

    particle_resized = cv2.resize(particle_img, (particle_size, particle_size))  # Resize image

    for particle in particles:
        if particle["opacity"] > 0:  
            frame = overlay_image(frame, particle_resized, particle["x"], particle["y"], particle["opacity"] / 255.0)

def extract_body_pixels(frame, body_box):
    """ Extracts the detected body region and breaks it into small pixel-based particles. """
    global glitch_particles, body_pixels, glitch_active, glitch_start_time

    x1, y1, x2, y2 = body_box  
    body_pixels = frame[y1:y2, x1:x2].copy()  # Extract body image

    glitch_particles = []  # Clear previous particles
    glitch_active = True  # Activate glitch effect
    glitch_start_time = time.time()  # Start glitch timer
    
    for i in range(0, body_pixels.shape[0], 5):  # Step for particle density
        for j in range(0, body_pixels.shape[1], 5):
            color = tuple(int(c) for c in body_pixels[i, j])  # Extract pixel color
            glitch_particles.append({
                "x": x1 + j, "y": y1 + i,
                "vx": 0, "vy": 0,
                "opacity": 255,
                "color": color  # Assign extracted color
            })

def apply_glitch_effect(frame, body_box):
    """ Applies a digital glitch effect to the body before dispersion. """
    x1, y1, x2, y2 = body_box  

    # Randomly shift horizontal & vertical lines to create a glitch
    for i in range(y1, y2, 10):
        shift = random.randint(-5, 5)  # Random horizontal shift
        frame[i:i+5, x1:x2] = np.roll(frame[i:i+5, x1:x2], shift, axis=1)

    for j in range(x1, x2, 10):
        shift = random.randint(-5, 5)  # Random vertical shift
        frame[y1:y2, j:j+5] = np.roll(frame[y1:y2, j:j+5], shift, axis=0)

    return frame

def dispersion_effect():
    """ Scatters particles outward to create a dispersion effect. """
    global glitch_particles
    for glitch_par in glitch_particles:
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(2, 6)

        glitch_par["vx"] = np.cos(angle) * speed  
        glitch_par["vy"] = np.sin(angle) * speed  

def update_glitch(frame, body_box, hands_together):
    """ Controls the glitch & dispersion effect when hands come together. """
    global glitch_particles, glitch_active, glitch_start_time

    if hands_together and body_box and not glitch_active:
        extract_body_pixels(frame, body_box)  # Extract pixels for glitch effect

    if glitch_active:
        elapsed_time = time.time() - glitch_start_time
        if elapsed_time < glitch_duration:
            frame = apply_glitch_effect(frame, body_box)  # Apply glitch effect
        else:
            glitch_active = False  # End glitch
            dispersion_effect()  # Start dispersion

    for glitch_par in glitch_particles:
        # Move particles outward
        glitch_par["x"] += glitch_par["vx"]
        glitch_par["y"] += glitch_par["vy"]
        glitch_par["opacity"] -= 5  # Smooth fade effect

        # Remove fully faded particles
        if glitch_par["opacity"] <= 0:
            glitch_particles.remove(glitch_par)

        # Keep particles within screen bounds
        glitch_par["x"] = np.clip(glitch_par["x"], 0, width)
        glitch_par["y"] = np.clip(glitch_par["y"], 0, height)

    return frame

def draw_glitch(frame):
    """ Draws pixel-based particles with fading effect. """
    for glitch_par in glitch_particles:
        if glitch_par["opacity"] > 0:
            color = glitch_par["color"]
            cv2.circle(frame, (int(glitch_par["x"]), int(glitch_par["y"])), 2, color, -1)
