import cv2
import numpy as np

# Screen size
projector_width = 1920  
projector_height = 1080  

cv2.namedWindow("Projector", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

video_path = "./video/myvideo.mp4"  
video_cap = cv2.VideoCapture(video_path)

def play_video_on_projector():
    global video_enabled

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
        if cv2.waitKey(1) & 0xFF == ord("q"): 
            return False
    return True  

def update_projector(frame, rain_enabled, video_enabled):
    if video_enabled:
        video_enabled = play_video_on_projector()
        return video_enabled
    
    projector_frame = np.full((projector_height, projector_width, 3), 0, dtype=np.uint8)
    from Effect.Particeles import draw_dispersion, draw_gravity_swirl_particles
    from Effect.Rain import initialize_rain, control_rain
    
    if rain_enabled:
        initialize_rain()
        control_rain(frame)
    draw_dispersion(projector_frame)
    draw_gravity_swirl_particles(projector_frame)
    
    gray_projector_frame = cv2.cvtColor(projector_frame, cv2.COLOR_BGR2GRAY)
    gray_projector_frame = cv2.cvtColor(gray_projector_frame, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Projector", gray_projector_frame)
    return video_enabled