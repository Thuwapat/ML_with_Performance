from ultralytics import YOLO
import numpy as np
import cv2

# โหลด YOLOv11
model_umbrella = YOLO("yolo11n.pt").to('cuda')

def detect_umbrella(frame):
    
    results = model_umbrella.predict(frame)

    umbrella_boxes = []
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()  
            for i, box in enumerate(boxes):
                if model_umbrella.names[int(class_ids[i])] == "umbrella":
                    umbrella_boxes.append(box)

    return umbrella_boxes

def add_rain_effect(frame):
    
    rain_layer = np.zeros_like(frame, dtype=np.uint8)
    h, w, _ = frame.shape

    num_drops = 100  
    for _ in range(num_drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        length = np.random.randint(5, 15)
        cv2.line(rain_layer, (x, y), (x, y + length), (255, 255, 255), 1)

    frame_with_rain = cv2.addWeighted(frame, 0.8, rain_layer, 0.2, 0)
    return frame_with_rain

def process_frame(frame):
   
    umbrellas = detect_umbrella(frame)

    if len(umbrellas) > 0:  
        frame = add_rain_effect(frame) 

    return frame


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = process_frame(frame)  
    cv2.imshow("Rain Effect with Umbrella Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
