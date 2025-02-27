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
    
    # ตัวแปรสำหรับป้องกันการกดซ้ำ
    last_key_time = 0
    key_cooldown = 0.2  # 200 มิลลิวินาที
    
    # ตัวแปรเก็บสถานะปุ่มที่กดล่าสุด
    pending_effect_change = None

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
        
        # แสดงเอฟเฟกต์ปัจจุบัน
        effect_text = f"Active Effect: {active_effect}"
        cv2.putText(frame, effect_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # ประมวลผลเอฟเฟกต์
        if active_effect == "gravity_swirl":
            effect_has_trail = False  # ไม่มีหางอนุภาค
            update_gravity_swirl_particles(body_box, elapsed_time, effect_has_trail)
        elif active_effect == "gravity/body":
            effect_has_trail = True  # มีหางอนุภาค
            update_gravity_swirl_particles(body_box, elapsed_time, effect_has_trail)
            update_body_energy_particles(body_box, elapsed_time)
        elif active_effect == "disperson":
            frame = update_dispersion(frame, body_box, body_keypoints)
        elif active_effect == "black_hole":
            frame = create_interstellar_black_hole(frame, hands_up)

            # สร้าง Frame สำหรับโปรเจคเตอร์
            projector_frame = np.zeros((projector_height, projector_width, 3), dtype=np.uint8)

            # วาดหลุมดำไปที่โปรเจคเตอร์
            projector_frame = create_interstellar_black_hole(projector_frame, hands_up)

            # ส่งไปแสดงผลที่หน้าต่างโปรเจคเตอร์
            cv2.imshow("Projector", projector_frame)


        video_enabled = update_projector(frame, rain_enabled, video_enabled, active_effect)


        # Show frame
        cv2.imshow("Demo", frame)
        
        # ตรวจจับคีย์บอร์ด - เพิ่มเวลารอให้นานขึ้น
        key = cv2.waitKey(5) & 0xFF  # เพิ่มเวลารอจาก 1 เป็น 5 มิลลิวินาที
        
        # ตรวจสอบว่าสามารถรับคำสั่งใหม่ได้หรือไม่
        if current_time - last_key_time > key_cooldown:
            if key == ord("q"):
                break
            elif key == ord("1"):
                pending_effect_change = "disperson"
                last_key_time = current_time
            elif key == ord("2"):
                pending_effect_change = "gravity_swirl"
                last_key_time = current_time
            elif key == ord("3"):
                pending_effect_change = "gravity/body"  
                last_key_time = current_time
            elif key == ord("4"): 
                video_enabled = not video_enabled
                last_key_time = current_time
            elif key == ord("5"):
                pending_effect_change = "rain"
                last_key_time = current_time
            elif key == ord("6"):
                pending_effect_change = "black_hole"
                last_key_time = current_time
            elif key == ord("0"):
                pending_effect_change = "none"
                last_key_time = current_time
            
        # ประมวลผลการเปลี่ยนเอฟเฟกต์
        if pending_effect_change is not None:
            clear_all_particles()
            active_effect = pending_effect_change
            
            # จัดการกรณีพิเศษสำหรับ rain
            if active_effect == "rain":
                rain_enabled = not rain_enabled
                
            pending_effect_change = None
            
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()