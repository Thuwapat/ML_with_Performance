from ultralytics import YOLO
import numpy as np
import cv2


model_post = YOLO("./Detection/yolo11n-pose.pt").to('cuda')
model_hand = YOLO("./Detection/hand_detection.pt").to('cuda')
model_object = YOLO("./Detection/yolo11n.pt").to('cuda')
model_phone = YOLO("./Detection/yolo11n.pt").to('cuda')  

def detect_phone(frame):
    results = model_phone.predict(frame)
    phone_boxes = []
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()  
            for i, box in enumerate(boxes):
                if int(class_ids[i]) == 67:  
                    phone_boxes.append(box)
    return phone_boxes

def add_rain_effect(frame):
    rain_layer = np.zeros_like(frame, dtype=np.uint8)
    h, w, _ = frame.shape
    num_fine = 500
    for _ in range(num_fine):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        length = np.random.randint(10, 30)
        thickness = 2
        wind = np.random.randint(-3, 4)
        rain_color = (220, 220, 255)
        cv2.line(rain_layer, (x, y), (x + wind, y + length), rain_color, thickness)
    frame_with_rain = cv2.addWeighted(frame, 0.7, rain_layer, 0.4, 0)
    return frame_with_rain

def process_frame(frame):
    phones = detect_phone(frame)  
    if len(phones) > 0: 
        frame = add_rain_effect(frame)  
    return frame

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = process_frame(frame)
    cv2.imshow("Rain Effect with Phone Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

