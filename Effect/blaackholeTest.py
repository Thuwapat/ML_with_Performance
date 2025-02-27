import cv2
import numpy as np
import torch
import time
from PIL import Image, ImageSequence
import os
from ultralytics import YOLO

# ตรวจสอบว่าใช้ GPU ได้หรือไม่
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# โหลดโมเดล YOLO เพียงครั้งเดียว
model_object = YOLO("./Detection/yolo11x.pt").to(DEVICE)

class PhoneDetector:
    def __init__(self, gif_path="Effect/bbtest.gif", conf_threshold=0.5):
        """
        กำหนดค่าเริ่มต้นสำหรับการตรวจจับโทรศัพท์และแสดง GIF
        """
        # ใช้โมเดลที่โหลดไว้แล้ว
        self.model = model_object
        self.conf_threshold = conf_threshold
        
        # โหลด GIF ที่จะแสดงเมื่อตรวจพบโทรศัพท์
        self.gif_path = gif_path
        self.gif_frames = self._load_gif_frames(gif_path)
        self.current_frame = 0
        self.showing_gif = False
        self.gif_start_time = 0
        self.gif_duration = 3  # แสดง GIF 3 วินาที
        
        # Class ID ของโทรศัพท์ใน COCO dataset (67 สำหรับโทรศัพท์มือถือ)
        self.phone_class_id = 67

    def _load_gif_frames(self, gif_path):
        """โหลดเฟรมของ GIF"""
        if not os.path.exists(gif_path):
            print(f"ไม่พบไฟล์ GIF ที่ {gif_path} - สร้างภาพพื้นหลังสีแดงแทน")
            placeholder = np.ones((300, 400, 3), dtype=np.uint8) * np.array([0, 0, 255], dtype=np.uint8)
            cv2.putText(placeholder, "พบโทรศัพท์!", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return [placeholder]
        
        gif = Image.open(gif_path)
        frames = []
        for frame in ImageSequence.Iterator(gif):
            frame_rgb = frame.convert('RGB')
            frame_array = np.array(frame_rgb)
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            frames.append(frame_bgr)
        return frames
    
    def detect_and_display(self, frame):
        """ตรวจจับโทรศัพท์และแสดง GIF"""
        if self.showing_gif:
            if time.time() - self.gif_start_time >= self.gif_duration:
                self.showing_gif = False
            else:
                self.current_frame = (self.current_frame + 1) % len(self.gif_frames)
                return self.gif_frames[self.current_frame]
        
        # ใช้ YOLOv8 ตรวจจับวัตถุ
        results = self.model.predict(frame, conf=self.conf_threshold, device=DEVICE)
        
        found_phone = False
        for detection in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = detection
            class_id = int(cls)
            
            if class_id == self.phone_class_id:
                found_phone = True
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'โทรศัพท์: {conf:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if found_phone and not self.showing_gif:
            self.showing_gif = True
            self.gif_start_time = time.time()
            self.current_frame = 0
            return self.gif_frames[self.current_frame]
        
        return frame

def main():
    detector = PhoneDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ไม่สามารถอ่านเฟรมจากกล้องได้")
            break
        
        processed_frame = detector.detect_and_display(frame)
        cv2.imshow('Phone Detector', processed_frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
