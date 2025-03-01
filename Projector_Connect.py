import cv2
import numpy as np
import random
import time
from PIL import Image, ImageDraw, ImageFont  # ใช้ PIL สำหรับวาดข้อความภาษาไทย

# Screen size สำหรับ Projector
projector_width = 1920  
projector_height = 1080  

# ตัวแปร global สำหรับข้อความใน Projector
projector_text_to_display = None
projector_text_start_time = 0
projector_text_duration = 3.0  # ระยะเวลาแสดงข้อความ (วินาที)

# ตั้งค่าหน้าต่าง Projector ให้เป็น fullscreen
is_fullscreen = True
cv2.namedWindow("Projector", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

video_path = "./video/myvideo.mp4"  
video_cap = cv2.VideoCapture(video_path)

def toggle_fullscreen():
    global is_fullscreen
    is_fullscreen = not is_fullscreen
    if is_fullscreen:
        cv2.setWindowProperty("Projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty("Projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

def play_video_on_projector():
    global video_enabled, is_fullscreen

    if not video_cap.isOpened():
        print("Can't play video")
        return False

    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  
            return False

        frame = cv2.resize(frame, (projector_width, projector_height))
        cv2.imshow("Projector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # ออกจากโปรแกรม
            return False
        elif key == 27:  # ESC - สลับโหมดเต็มหน้าจอ
            toggle_fullscreen()
    return True  

def put_centered_text(frame, text, font_path, font_size, color):
    """
    ฟังก์ชันช่วยวาดข้อความ (รองรับ Unicode) ให้อยู่กลางจอ Projector โดยใช้ PIL
    ใช้ font.getbbox() คำนวณขนาดข้อความแทน getsize()
    """
    # แปลง OpenCV image (BGR) เป็น PIL image (RGB)
    cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    draw = ImageDraw.Draw(pil_im)
    # โหลดฟอนต์ที่รองรับภาษาไทย (ตรวจสอบ path ให้ถูกต้อง)
    font = ImageFont.truetype("Front/THSarabun.ttf", font_size)
    # ใช้ getbbox() เพื่อคำนวณขนาดของข้อความ
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (pil_im.width - text_width) // 2
    y = (pil_im.height - text_height) // 2
    draw.text((x, y), text, font=font, fill=color)  # color ใน PIL เป็นแบบ RGB
    # แปลงกลับเป็น OpenCV image (BGR)
    cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    return cv2_im_processed

def update_projector(hands_up, rain_enabled, video_enabled, active_effect, lightning_effect=False):
    global projector_text_to_display, projector_text_start_time, projector_text_duration
    if video_enabled:
        video_enabled = play_video_on_projector()
        return video_enabled
    
    projector_frame = np.full((projector_height, projector_width, 3), 0, dtype=np.uint8)

    # รวมเอฟเฟกต์ต่างๆ เข้าไปในหน้าจอ Projector
    from Effect.Particeles import draw_dispersion, draw_gravity_swirl_particles
    from Effect.Rain import add_rain_effect

    if rain_enabled:
        projector_frame = add_rain_effect(projector_frame)
        
    if lightning_effect: 
        flash_intensity = random.randint(80, 150) 
        lightning_frame = np.full_like(projector_frame, flash_intensity)
        projector_frame = cv2.addWeighted(projector_frame, 0.8, lightning_frame, 0.5, 0)

    if active_effect == "black_hole":
        from Effect.Interstellar_blackHole import create_interstellar_black_hole
        projector_frame = create_interstellar_black_hole(projector_frame, hands_up)
    if active_effect == "firework":
        from Effect.Firework import firework_effect, draw_firework
        firework_effect(projector_width, projector_height)  # อัปเดตตำแหน่งของพลุ
        draw_firework(projector_frame)  # วาดพลุบน Projector
    draw_dispersion(projector_frame)         # วาดเอฟเฟกต์กระจายพลังงาน
    draw_gravity_swirl_particles(projector_frame)  # วาดอนุภาคหมุนรอบตัว
    
    # วาดข้อความลงใน projector_frame หากมีการกำหนดข้อความ
    if projector_text_to_display is not None:
        current_time = time.time()
        if current_time - projector_text_start_time < projector_text_duration:
            # ใช้ฟังก์ชัน put_centered_text ที่รองรับ Unicode ด้วย PIL
            # ระบุ path ของฟอนต์ที่รองรับภาษาไทย เช่น "THSarabun.ttf" หรือแก้ path ให้ถูกต้อง
            projector_frame = put_centered_text(projector_frame, projector_text_to_display, "THSarabun.ttf", 64, (0, 255, 255))
        else:
            projector_text_to_display = None

    cv2.imshow("Projector", projector_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        return False
    elif key == 27:
        toggle_fullscreen()
    
    return video_enabled
