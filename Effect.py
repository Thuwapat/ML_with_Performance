import random
import cv2
import numpy as np  

# Screen size
width, height = 640, 480

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
    particles = [{"x": random.randint(0, width), "y": random.randint(0, height), "vx": 0, "vy": 0, "size": random.randint(1, 3)} for _ in range(num_particles)]

def update_particles(hand_center, hand_open):
    global particles

    if hand_center is None:
        return  # No hand detected, particles move freely

    for particle in particles:
        # Compute direction vector from hand to particle
        direction_x = particle["x"] - hand_center[0]
        direction_y = particle["y"] - hand_center[1]
        distance = max(np.sqrt(direction_x**2 + direction_y**2), 1)  # Avoid division by zero

        # Normalize direction
        direction_x /= distance
        direction_y /= distance

        # Apply movement behavior based on hand gesture
        speed = 3 if hand_open else -3  # If open palm, move outward; if fist, move inward
        particle["vx"] += direction_x * speed + random.uniform(-0.5, 0.5)
        particle["vy"] += direction_y * speed + random.uniform(-0.5, 0.5)

        # Apply velocity with some friction to keep motion smooth
        particle["x"] += particle["vx"]
        particle["y"] += particle["vy"]
        particle["vx"] *= 0.9  # Friction effect
        particle["vy"] *= 0.9

        # Keep particles within screen bounds
        particle["x"] = np.clip(particle["x"], 0, 640)
        particle["y"] = np.clip(particle["y"], 0, 480)

def draw_particles(frame):
    for particle in particles:
        cv2.circle(frame, (int(particle["x"]), int(particle["y"])), 3, (255, 255, 255), -1)
