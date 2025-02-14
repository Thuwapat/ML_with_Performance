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
num_particles = 200
particles = []


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

def heart_shape(index, scale=10):
    t = index * (2 * np.pi / num_particles)
    x = scale * (16 * np.sin(t)**3)
    y = -scale * (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))  # Inverted Y for OpenCV
    return x, y

## If palm is open, particles burst outward and fade away
## If handful is made, particles arrange into a shape
def update_particles(hand_center, hand_open, handful): 
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
            particle["vx"] += direction_x * speed + random.uniform(-1, 1)
            particle["vy"] += direction_y * speed + random.uniform(-1, 1)

            particle["opacity"] -= 10  
            if particle["opacity"] <= 0 and not cooldown_active:
                last_burst_time = current_time  
                reset_particles_to_borders()

        if handful:
            target_x, target_y = heart_shape(i, scale=10)
            target_x += hand_x  
            target_y += hand_y

            particle["vx"] += (target_x - particle["x"]) * 0.1  
            particle["vy"] += (target_y - particle["y"]) * 0.1  

        particle["x"] += particle["vx"]
        particle["y"] += particle["vy"]
        particle["vx"] *= 0.9  
        particle["vy"] *= 0.9

        particle["x"] = np.clip(particle["x"], 0, width)
        particle["y"] = np.clip(particle["y"], 0, height)

def draw_particles(frame):
    for particle in particles:
        opacity = max(0, particle["opacity"])  
        if opacity > 0:  
            cv2.circle(frame, (int(particle["x"]), int(particle["y"])), particle["size"], (170, 9, 78), -1)
