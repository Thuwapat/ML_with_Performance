import cv2
import time
from Pose_estimation import get_shoulder_x
from Effect import create_snowflakes, update_snowflakes, draw_snowflakes
from Utileize import calculate_horizontal_angle

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize snowflakes
create_snowflakes()

# Track previous shoulder angle
last_angle = None
spin_detected = False
last_spin_time = 0  # To track when spinning stops

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get shoulder X-coordinates and keypoints
    left_shoulder_x, right_shoulder_x, keypoints = get_shoulder_x(frame)

    # Draw keypoints on frame
    if keypoints is not None:
        for x, y in keypoints:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green dots for keypoints

    if left_shoulder_x is not None and right_shoulder_x is not None:
        # Compute horizontal angle
        current_angle = calculate_horizontal_angle(left_shoulder_x, right_shoulder_x)

        if last_angle is not None:
            angle_change = abs(current_angle - last_angle)

            # Detect fast spinning motion
            if angle_change > 20:
                spin_detected = True
                last_spin_time = time.time()  # Reset spin timer

        last_angle = current_angle  # Update last angle

    # Stop snow if no spin detected 
    if time.time() - last_spin_time > 3:
        spin_detected = False  # Stop snow effect

    # If spin detected, activate snow effect
    if spin_detected:
        update_snowflakes()
        draw_snowflakes(frame)

    # Show frame
    cv2.imshow("YOLO Pose Estimation with Horizontal Spin-Triggered Snow Effect", frame)

cap.release()
cv2.destroyAllWindows()
