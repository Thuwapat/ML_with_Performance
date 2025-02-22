import cv2
import time
from Detection.Get_Var import *
from Effect.Particeles import *
from Utileize import calculate_horizontal_angle
import pyvirtualcam

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    # Time tracking for fade effect
    fade_duration = 30  # in seconds
    fade_factor = 1  # Ensure fade_factor is initialized
    start_fade_time = time.time()
    initialize_particles()

    with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
        prev_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (width, height))

            left_shoulder_x, right_shoulder_x, keypoint = get_post_keypoint(frame)
            hand_boxes, hand_keypoints, left_hand, right_hand = get_hand_keypoint(frame)
            left_hand, right_hand, handful, hand_center, hand_open, hands_together = detect_hand(frame)
            body_box = detect_body(frame)
            
            # Smooth frame rate control
            current_time = time.time()
            elapsed_time = current_time - prev_time
            prev_time = current_time  

            if get_dispersion_status():
                if start_fade_time is None:  
                    start_fade_time = current_time  # Start fade effect once

                fade_elapsed_time = current_time - start_fade_time
                fade_factor = max(0, 1 - (fade_elapsed_time / fade_duration))

                # Convert frame to black using a blend effect
                black_background = np.zeros_like(frame)  
                frame = cv2.addWeighted(frame, fade_factor, black_background, 1 - fade_factor, 0)
            
            # If screen is fully black, disable glitch effect
            apply_glitch_effect = fade_factor > 0

            # Draw Body keypoints on frame
            if keypoint is not None:
                for x, y in keypoint:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green dots for keypoints

             # Draw hand marker (for visualization)
            if hand_center is not None:
                color = (0, 255, 0) if hand_open else (0, 0, 255)  # Green for open, Red for fist
                cv2.circle(frame, hand_center, 20, color, 2)

            # Draw Hand box on frame
            for box in hand_boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #
            #Draw **hand keypoints** (Red dots)
            for keypoint in hand_keypoints:
                for x, y in keypoint:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            update_gravity_swirl_particles(left_hand, right_hand, hand_center, hand_open, handful, elapsed_time)
            update_body_energy_particles(body_box, hand_center, hand_open, elapsed_time)
            draw_gravity_swirl_particles(frame)

            if apply_glitch_effect:
                frame = update_glitch(frame, body_box, hands_together)
                draw_glitch(frame)

            # Gray Filter
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

            # Show frame
            cv2.imshow("Demo", frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                # Apply effects as per your script...

            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert for virtual cam
            cam.send(frame)
            cam.sleep_until_next_frame()
            
        cap.release()
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()