import cv2
import time
from Detection.Get_Var import *
from Effect.Particeles import *
from Effect.Interstellar_blackHole import create_interstellar_black_hole
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
    lightning_effect = False
    lightning_timer = 0

    umbrella_last_seen_time = None  # ✅ เก็บเวลาที่พบร่มล่าสุด
    rain_delay = 3  # ✅ ดีเลย์ 3 วินาทีก่อนปิดฝน

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
        elif active_effect == "rain":
            if len(umbrellas) > 0:
                rain_enabled = True
                umbrella_last_seen_time = current_time
                if lightning_timer <= 0:
                    lightning_effect = True
                    lightning_timer = random.randint(5, 10)
                else:
                    if umbrella_last_seen_time and (current_time - umbrella_last_seen_time > rain_delay):
                        lightning_effect = False
                        lightning_timer -= 1  # ✅ ลดค่าตัวจับเวลา
            else:
                rain_enabled = False
                lightning_effect = False
            # สร้าง Frame สำหรับโปรเจคเตอร์
            projector_frame = np.zeros((projector_height, projector_width, 3), dtype=np.uint8)

            # วาดหลุมดำไปที่โปรเจคเตอร์
            projector_frame = create_interstellar_black_hole(projector_frame, hands_up)

            # ส่งไปแสดงผลที่หน้าต่างโปรเจคเตอร์
            cv2.imshow("Projector", projector_frame)


        video_enabled = update_projector(frame, rain_enabled, video_enabled, active_effect, lightning_effect)


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
            elif key == ord("7"):  
                video_enabled = True
                while video_enabled:
                    video_enabled = play_gif_on_projector()
                    last_key_time = current_time  
            elif key == ord("0"):
                pending_effect_change = "none"
                last_key_time = current_time
            
        # ประมวลผลการเปลี่ยนเอฟเฟกต์
        if pending_effect_change is not None:
            clear_all_particles()
            active_effect = pending_effect_change
            
            pending_effect_change = None
            
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()