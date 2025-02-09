import numpy as np
import cv2
import random
import torch
from ultralytics import YOLO

# ✅ Load YOLO Hand Detection Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_hand = YOLO("hand_detection.pt").to(device)  # Make sure to have this model!

# ✅ Initialize Webcam
cap = cv2.VideoCapture(0)

# ✅ Particle properties
num_particles = 200
particles = [{"x": random.randint(100, 540), "y": random.randint(100, 380), "vx": 0, "vy": 0} for _ in range(num_particles)]

def detect_hand(frame):
    """
    Uses YOLO to detect hands and returns their center position + open/closed state.
    """
    results = model_hand.predict(frame)

    hand_center, hand_open = None, False  # Default values if no hand is detected

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:  # Ensure detection exists
            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else []

            # ✅ Use first detected hand (for now)
            x1, y1, x2, y2 = boxes[0][:4]
            hand_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))  # Center of bounding box

            # ✅ Check if fingers are spread (i.e., open palm)
            if len(keypoints) > 0:
                finger_spread = np.mean(keypoints[0][:, 1]) > y1 + (y2 - y1) * 0.5  # Simple spread detection
                hand_open = finger_spread  # If fingers are high above palm, assume open

    return hand_center, hand_open  # Return position + open/closed state

def update_particles(hand_center, hand_open):
    """
    Moves particles outward when the palm is open, inward when a fist is made.
    """
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
    """
    Draws the floating particles on the screen.
    """
    for particle in particles:
        cv2.circle(frame, (int(particle["x"]), int(particle["y"])), 3, (255, 255, 255), -1)

# ✅ Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect for natural interaction

    # ✅ Detect hand position & gesture
    hand_center, hand_open = detect_hand(frame)

    # ✅ Update and draw particles
    update_particles(hand_center, hand_open)
    draw_particles(frame)

    # ✅ Draw hand marker (for visualization)
    if hand_center is not None:
        color = (0, 255, 0) if hand_open else (0, 0, 255)  # Green for open, Red for fist
        cv2.circle(frame, hand_center, 20, color, 2)

    # ✅ Display the frame
    cv2.imshow("YOLO Hand Tracking + Particle Interaction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
