import cv2
import time
from Get_Var import *
from Effect import *
from Utileize import calculate_horizontal_angle
import pyvirtualcam

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Initialize snowflakes
    #create_snowflakes()
    create_particles()

    # Track previous shoulder angle
    last_angle = None
    spin_detected = False
    last_spin_time = 0  # To track when spinning stops

    with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
        prev_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (width, height))
            # Get shoulder and hand X-coordinates and keypoints
            #left_shoulder_x, right_shoulder_x, keypoints = get_post_keypoint(frame)

            #hand_boxes, hand_keypoints, left_hand, right_hand = get_hand_keypoint(frame)

            left_hand, right_hand, handful, hand_center, hand_open, hands_together = detect_hand(frame)
            body_box = detect_body(frame)
            
            # Smooth frame rate control
            current_time = time.time()
            elapsed_time = current_time - prev_time
            prev_time = current_time  

            # Draw Body keypoints on frame
            #if keypoints is not None:
            #    for x, y in keypoints:
            #        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green dots for keypoints

             # Draw hand marker (for visualization)
            #if hand_center is not None:
            #    color = (0, 255, 0) if hand_open else (0, 0, 255)  # Green for open, Red for fist
            #    cv2.circle(frame, hand_center, 20, color, 2)

            # Draw Hand box on frame
            #for box in hand_boxes:
            #    x1, y1, x2, y2 = map(int, box[:4])
            #    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #
            # # Draw **hand keypoints** (Red dots)
            #for keypoint in hand_keypoints:
            #    for x, y in keypoint:
            #        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)


            #if left_shoulder_x is not None and right_shoulder_x is not None:
                # Compute horizontal angle
            #    current_angle = calculate_horizontal_angle(left_shoulder_x, right_shoulder_x)

            #    if last_angle is not None:
            #        angle_change = abs(current_angle - last_angle)

                    # Detect fast spinning motion
            #        if angle_change > 20:
            #            spin_detected = True
            #            last_spin_time = time.time()  # Reset spin timer

            #    last_angle = current_angle  # Update last angle

            # Stop snow if no spin detected 
            #if time.time() - last_spin_time > 3:
            #    spin_detected = False  # Stop snow effect

            # If spin detected, activate snow effect
            #if spin_detected:
            #    update_snowflakes()
            #    draw_snowflakes(frame)
            #else:
            update_particles(hand_center, hand_open, handful, elapsed_time)
            draw_particles(frame)
            frame = update_glitch(frame, body_box, hands_together)
            draw_glitch(frame)
            

            # Show frame
            cv2.imshow("Demo", frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                # Apply effects as per your script...

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert for virtual cam
            cam.send(frame)
            cam.sleep_until_next_frame()
            
        cap.release()
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()