import cv2
import numpy as np

# Screen size
projector_width = 3840  
projector_height = 2160  

cv2.namedWindow("Projector", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def update_projector():
    projector_frame = np.zeros((projector_height, projector_width, 3), dtype=np.uint8)
    from Effect.Particeles import draw_glitch, draw_gravity_swirl_particles
    draw_glitch(projector_frame)  # วาดอนุภาค Dispersion บนโปรเจคเตอร์
    draw_gravity_swirl_particles(projector_frame) 
    gray_projector_frame = cv2.cvtColor(projector_frame, cv2.COLOR_BGR2GRAY)
    gray_projector_frame = cv2.cvtColor(gray_projector_frame, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels
    cv2.imshow("Projector", gray_projector_frame)