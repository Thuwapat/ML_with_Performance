import random
import cv2

# Screen size
width, height = 640, 480

# Snowflake storage
num_flakes = 100
snowflakes = []

# Gen snow with Random positions
def create_snowflakes():
    global snowflakes
    snowflakes = [{"x": random.randint(0, width), "y": random.randint(0, height), "size": random.randint(2, 6)} for _ in range(num_flakes)]

# Make snow move downward and reste when reach bottom
def update_snowflakes():
    global snowflakes
    for flake in snowflakes:
        flake["y"] += random.randint(1, 4)  # Snowfall speed
        flake["x"] += random.choice([-1, 0, 1])  # Slight sideways drift
        if flake["y"] > height:
            flake["y"] = 0  # Reset to top
            flake["x"] = random.randint(0, width)

# Just Draw to each frame
def draw_snowflakes(frame):
    for flake in snowflakes:
        cv2.circle(frame, (flake["x"], flake["y"]), flake["size"], (255, 255, 255), -1)
