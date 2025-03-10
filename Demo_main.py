import cv2
import time
from Detection.Get_Var import *
from Effect.Particeles import *
from Effect.Interstellar_blackHole import create_interstellar_black_hole
from Utileize import calculate_shoulder_speed
from Projector_Connect import *


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set camera width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set camera height

    initialize_particles()
    
    prev_time = time.time()

    active_effect = "none"
    rain_enabled = False
    video_enabled = False
    effect_has_trail = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
            # รับค่าจาก get_post_keypoint() (6 ค่า)
        left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, all_keypoints = get_post_keypoint(frame)

        body_keypoints = [left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist]

        hand_boxes, hand_keypoints, left_hand, right_hand = get_hand_keypoint(frame)
        left_hand, right_hand, handful, hand_center, hand_open, hands_together = detect_hand(frame)
        hands_up = is_hands_up(left_shoulder, right_shoulder, left_wrist, right_wrist)
        body_box = detect_body(frame)
        umbrellas = detect_umbrella(frame)

        # Smooth frame rate control
        current_time = time.time()
        elapsed_time = current_time - prev_time
        prev_time = current_time 
            
        # Draw Body keypoints on frame
        if all_keypoints is not None:
            for x, y in all_keypoints:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green dots for keypoints
         # Draw hand marker (for visualization)
        if hand_center is not None:
            color = (0, 255, 0) if hand_open else (0, 0, 255)  # Green for open, Red for fist
            cv2.circle(frame, hand_center, 20, color, 2)
        # Draw Hand box on frame
        for box in hand_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #Draw **hand keypoints** (Red dots)
        for keypoint in hand_keypoints:
            for x, y in keypoint:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        # Draw Umbella
        for x1, y1, x2, y2 in umbrellas:
           cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # กล่องสีน้ำเงิน

        if active_effect == "gravity_swirl":
            effect_has_trail = False  # ไม่มีหางอนุภาค
            update_gravity_swirl_particles(body_box, elapsed_time, effect_has_trail)
        elif active_effect == "gravity/body":
            effect_has_trail = True  # มีหางอนุภาค
            update_gravity_swirl_particles(body_box, elapsed_time, effect_has_trail)
            update_body_energy_particles(body_box, elapsed_time)
        elif active_effect == "disperson":
            frame = update_dispersion(frame, body_box, body_keypoints)
        elif active_effect == "rain":
            rain_enabled = not rain_enabled
        elif active_effect == "black_hole":
            frame = create_interstellar_black_hole(frame, hands_up)

        video_enabled = update_projector(frame, rain_enabled, video_enabled)

        # Show frame
        cv2.imshow("Demo", frame)
        # ตรวจจับคีย์บอร์ด
        key = cv2.waitKey(1) & 0xFF  

        if key == ord("q"):
            break
        elif key == ord("1"):
            clear_all_particles()
            active_effect = "disperson"
        elif key == ord("2"):
            clear_all_particles()
            active_effect = "gravity_swirl"
        elif key == ord("3"):
            clear_all_particles()
            active_effect = "gravity/body"  
        elif key == ord("4"): 
            video_enabled = not video_enabled
        elif key == ord("5"):
            clear_all_particles()
            active_effect = "rain"
        elif key == ord("6"):
            active_effect = "black_hole"
        elif key == ord("0"):
            clear_all_particles()
            active_effect = "none"
            
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()