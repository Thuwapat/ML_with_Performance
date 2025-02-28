import cv2
import numpy as np
import random 
from PIL import Image, ImageSequence

# Screen size
projector_width = 1920  
projector_height = 1080  

# เริ่มต้นด้วยหน้าต่างปกติ (ไม่ใช่เต็มหน้าจอ)
is_fullscreen = True
cv2.namedWindow("Projector", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

video_path = "./video/myvideo.mp4"  
video_cap = cv2.VideoCapture(video_path)

gif_path = "./Effect/myblackhole.gif"
gif_frames = []

def load_gif():
    global gif_frames
    gif_frames = []
    gif = Image.open(gif_path)
    for frame in ImageSequence.Iterator(gif):
        frame = frame.convert("RGB")
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gif_frames.append(frame)

def play_gif_on_projector():
    global gif_frames
    if not gif_frames:
        load_gif()

    for frame in gif_frames:
        frame = cv2.resize(frame, (projector_width // 2, projector_height // 2))  # ✅ ปรับขนาด GIF
        x_offset = (projector_width - frame.shape[1]) // 2
        y_offset = (projector_height - frame.shape[0]) // 2

        projector_frame = np.full((projector_height, projector_width, 3), 0, dtype=np.uint8)
        projector_frame[y_offset:y_offset+frame.shape[0], x_offset:x_offset+frame.shape[1]] = frame

        cv2.imshow("Projector", projector_frame)

        key = cv2.waitKey(100) & 0xFF  # ✅ ควบคุมความเร็วของ GIF
        if key == ord("q") or key == ord("0"):  # ✅ หยุดเล่น GIF
            return False
    return True

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

def update_projector(frame, rain_enabled, video_enabled, active_effect, lightning_effect=False):
    if video_enabled:
        video_enabled = play_video_on_projector()
        return video_enabled
    
    projector_frame = np.full((projector_height, projector_width, 3), 0, dtype=np.uint8)

    # รวมเอฟเฟกต์ต่างๆ เข้าไปในหน้าจอโปรเจคเตอร์
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
        projector_frame = create_interstellar_black_hole(projector_frame, hands_up=True)
    
    draw_dispersion(projector_frame)  # วาดเอฟเฟกต์กระจายพลังงาน
    draw_gravity_swirl_particles(projector_frame)  # วาดอนุภาคหมุนรอบตัว
    
    # แสดงผลที่หน้าจอโปรเจคเตอร์
    cv2.imshow("Projector", projector_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        return False
    elif key == 27:
        toggle_fullscreen()
    
    return video_enabled

