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
    apply_glitch_effect = True

    initialize_particles()
    
    prev_time = time.time()

    active_effect = "all"  # ค่าเริ่มต้น: แสดงทุกเอฟเฟค

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        left_shoulder_x, right_shoulder_x, keypoint = get_post_keypoint(frame)
        hand_boxes, hand_keypoints, left_hand, right_hand = get_hand_keypoint(frame)
        left_hand, right_hand, handful, hand_center, hand_open, hands_together = detect_hand(frame)
        body_box = detect_body(frame)
        #umbrellas = detect_umbrella(frame)

        # Smooth frame rate control
        current_time = time.time()
        elapsed_time = current_time - prev_time
        prev_time = current_time 

        if get_dispersion_status():
           dispersion_effect(body_box)  # ทำให้อนุภาค Dispersion ยังคงเคลื่อนที่ออกจากจอ
           update_projector(frame)
           apply_glitch_effect = False  # ปิด Glitch Effect อย่างสมบูรณ์
            
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
        #Draw **hand keypoints** (Red dots)
        for keypoint in hand_keypoints:
            for x, y in keypoint:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        # Draw Umbella
        #for x1, y1, x2, y2 in umbrellas:
        #    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # กล่องสีน้ำเงิน

        #shoulder_speed = calculate_shoulder_speed(left_shoulder_x, right_shoulder_x, current_time)
        #frame = create_interstellar_black_hole(frame, shoulder_speed)
        if active_effect == "gravity_swirl":
            update_gravity_swirl_particles(left_hand, right_hand, hand_center, hand_open, handful, elapsed_time)
            update_body_energy_particles(body_box, hand_center, hand_open, elapsed_time)
        elif active_effect == "disperson":
            frame = update_glitch(frame, body_box, hands_together)
      

        if apply_glitch_effect and not get_dispersion_status():
            frame = update_glitch(frame, body_box, hands_together)
        
        
        update_projector(frame)

        # Show frame
        cv2.imshow("Demo", frame)
        # ตรวจจับคีย์บอร์ด
        key = cv2.waitKey(1) & 0xFF  

        if key == ord("q"):
            break
        elif key == ord("1"):
            active_effect = "disperson"
        elif key == ord("2"):
            active_effect = "gravity_swirl"
            
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()