import cv2
import time
import numpy as np
import random
from Detection.Get_Var import *
from Effect.Particeles import *
import Projector_Connect  # import โมดูล Projector_Connect โดยตรง

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Set camera width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)    # Set camera height

    initialize_particles()
    
    prev_time = time.time()

    active_effect = "none"
    rain_enabled = False
    video_enabled = False
    effect_has_trail = False
    lightning_effect = False
    lightning_timer = 0

    umbrella_last_seen_time = None  # เก็บเวลาที่พบร่มล่าสุด
    rain_delay = 3  # ดีเลย์ 3 วินาทีก่อนปิดฝน
    
    # ตัวแปรสำหรับป้องกันการกดซ้ำ
    last_key_time = 0
    key_cooldown = 0.2  # 200 มิลลิวินาที
    
    # ตัวแปรเก็บสถานะการเปลี่ยนเอฟเฟกต์ (ยังไม่เปลี่ยนจนกว่าข้อความจะแสดงครบ)
    pending_effect_change = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        
        # รับค่าจาก get_post_keypoint() (6 ค่า)
        left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, all_keypoints = get_post_keypoint(frame)
        body_keypoints = [left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist]
        hands_up = is_hands_up(left_shoulder, right_shoulder, left_wrist, right_wrist)
        body_box = detect_body(frame)
        umbrellas = detect_umbrella(frame)

        current_time = time.time()
        elapsed_time = current_time - prev_time
        prev_time = current_time 
            
        # วาด Body keypoints ลงบน Demo window
        if all_keypoints is not None:
            for x, y in all_keypoints:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        
        # วาดกรอบวัตถุ "umbrella"
        for x1, y1, x2, y2 in umbrellas:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        
        # แสดงข้อความเอฟเฟกต์ปัจจุบันบน Demo window (สำหรับ debugging)
        effect_text = f"Active Effect: {active_effect}"
        cv2.putText(frame, effect_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # ประมวลผลเอฟเฟกต์ตาม active_effect
        if active_effect == "gravity_swirl":
            effect_has_trail = False  # ไม่มีหางอนุภาค
            update_gravity_swirl_particles(body_box, elapsed_time, effect_has_trail)
        elif active_effect == "gravity/body":
            effect_has_trail = True   # มีหางอนุภาค
            update_gravity_swirl_particles(body_box, elapsed_time, effect_has_trail)
            update_body_energy_particles(body_box, elapsed_time)
        elif active_effect == "disperson":
            frame = update_dispersion(frame, body_box, body_keypoints)
        elif active_effect == "black_hole":
            pass
        elif active_effect == "firework":
            pass  # ให้ Projector จัดการเอฟเฟกต์เอง
        elif active_effect == "rain":
            if len(umbrellas) > 0:
                rain_enabled = True
                umbrella_last_seen_time = current_time
                if lightning_timer <= 0:
                    lightning_effect = True
                    lightning_timer = random.randint(5, 10)
                else:
                    lightning_effect = False
                    lightning_timer -= 1
            else:
                if umbrella_last_seen_time and (current_time - umbrella_last_seen_time > rain_delay):
                    rain_enabled = False
                    lightning_effect = False
            
            # สำหรับ rain effect สร้าง frame เบื้องต้นให้ Projector
            projector_frame = np.zeros((Projector_Connect.projector_height, Projector_Connect.projector_width, 3), dtype=np.uint8)
            cv2.imshow("Projector", projector_frame)

        # อัปเดตหน้าจอ Projector
        video_enabled = Projector_Connect.update_projector(hands_up, rain_enabled, video_enabled, active_effect, lightning_effect)

        # แสดงผล Demo window
        cv2.imshow("Demo", frame)
        
        # ตรวจจับคีย์บอร์ด
        key = cv2.waitKey(5) & 0xFF
        
        if current_time - last_key_time > key_cooldown:
            if key == ord("q"):
                break
            elif key == ord("1"):
                pending_effect_change = "black_hole"
                last_key_time = current_time
            elif key == ord("2"):
                # เมื่อกดเลข 2 ตั้งค่าข้อความใน Projector แล้วรอให้ครบช่วงเวลา ก่อนเปลี่ยนเอฟเฟกต์
                Projector_Connect.projector_text_to_display = "เรามักจะรู้สึกเหมือนโดนดูดอยู่ในหลุมดำตลอดเวลา"
                Projector_Connect.projector_text_start_time = current_time
                pending_effect_change = "gravity_swirl"
                last_key_time = current_time
            elif key == ord("3"):
                pending_effect_change = "rain"
                last_key_time = current_time
            elif key == ord("4"):
                Projector_Connect.projector_text_to_display = "เฝ้าทะนุถนอมความรู้สึกที่มีต่ออีกฝ่าย"
                Projector_Connect.projector_text_start_time = current_time
                pending_effect_change = "firework"
                last_key_time = current_time
            elif key == ord("5"):
                pending_effect_change = "disperson"
                last_key_time = current_time
            elif key == ord("6"):
                Projector_Connect.projector_text_to_display = "แต่เราเป็นฝ่ายแตกสลายนับไม่ถ้วน"
                Projector_Connect.projector_text_start_time = current_time
                pending_effect_change = "gravity/body"
                last_key_time = current_time
            elif key == ord("7"):
                video_enabled = not video_enabled
                last_key_time = current_time
            elif key == ord("0"):
                pending_effect_change = "none"
                last_key_time = current_time
            
        # หากมี pending_effect_change อยู่ ให้รอจนข้อความแสดงครบก่อนเปลี่ยน effect
        if pending_effect_change is not None:
            if Projector_Connect.projector_text_to_display is not None:
                if current_time - Projector_Connect.projector_text_start_time >= Projector_Connect.projector_text_duration:
                    clear_all_particles()
                    active_effect = pending_effect_change
                    pending_effect_change = None
            else:
                clear_all_particles()
                active_effect = pending_effect_change
                pending_effect_change = None
            
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
