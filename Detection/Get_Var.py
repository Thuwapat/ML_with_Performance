from ultralytics import YOLO
import numpy as np
import cv2
import torch

# ตรวจสอบว่ามี GPU หรือไม่
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# โหลดโมเดล YOLO เพียงครั้งเดียว
model_post = YOLO("./Detection/yolo11n-pose.pt").to(DEVICE)
model_hand = YOLO("./Detection/hand_detection.pt").to(DEVICE)
model_object = YOLO("./Detection/yolo11x.pt").to(DEVICE)
model_seg = YOLO("./Detection/yolo11x-seg.pt").to(DEVICE)

def get_post_keypoint(frame):
    results = model_post.predict(frame, device=DEVICE)
    
    if not results:
        return None, None, []
    
    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()
            if keypoints.shape[1] >= 7:
                left_shoulder_x = keypoints[0][5][0] if len(keypoints[0]) > 5 else None
                right_shoulder_x = keypoints[0][6][0] if len(keypoints[0]) > 6 else None
                return left_shoulder_x, right_shoulder_x, keypoints[0]
    
    return None, None, []

def get_hand_keypoint(frame):
    results = model_hand.predict(frame, device=DEVICE)
    
    hand_boxes, hand_keypoints = [], []
    left_hand, right_hand = None, None
    
    for result in results:
        if result.boxes is not None:
            hand_boxes.extend(result.boxes.xyxy.cpu().numpy())
        if result.keypoints is not None:
            hand_keypoints.extend(result.keypoints.xy.cpu().numpy())

    if len(hand_keypoints) > 0:
        left_hand = hand_keypoints[0][:2] if len(hand_keypoints[0]) >= 2 else None
    if len(hand_keypoints) > 1:
        right_hand = hand_keypoints[1][:2] if len(hand_keypoints[1]) >= 2 else None
    
    return hand_boxes, hand_keypoints, left_hand, right_hand

def detect_hand(frame):
    results = model_hand.predict(frame, device=DEVICE)
    
    left_hand, right_hand = None, None
    handful, hand_open, hands_together = False, False, False
    hand_center = None
    
    height, width, _ = frame.shape
    
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None and len(result.keypoints) > 0 else []
            
            if len(boxes) >= 1:
                left_hand = (int((boxes[0][0] + boxes[0][2]) / 2), int((boxes[0][1] + boxes[0][3]) / 2))
            if len(boxes) >= 2:
                right_hand = (int((boxes[1][0] + boxes[1][2]) / 2), int((boxes[1][1] + boxes[1][3]) / 2))
            
            hand_center = left_hand if right_hand is None else right_hand
            
            if len(keypoints) > 0 and keypoints[0].shape[0] >= 21:
                palm_y = keypoints[0][0, 1]
                finger_tips_y = [keypoints[0][i, 1] for i in [8, 12, 16, 20]]
                avg_finger_tip_y = np.mean(finger_tips_y)
                threshold = 30
                hand_open = avg_finger_tip_y < (palm_y - threshold)
                handful = not hand_open
            
            if left_hand and right_hand:
                distance = np.sqrt((left_hand[0] - right_hand[0])**2 + (left_hand[1] - right_hand[1])**2)
                hands_together = distance < width * 1
    
    return left_hand, right_hand, handful, hand_center, hand_open, hands_together

def detect_body(frame):
    results = model_object.predict(frame, device=DEVICE)
    
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            x1, y1, x2, y2 = map(int, result.boxes.xyxy.cpu().numpy()[0])
            return x1, y1, x2, y2
    
    return None

def get_body_mask(frame):
    results = model_seg.predict(frame, device=DEVICE)
    
    if results:
        combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        for result in results:
            if result.masks is not None:
                for mask_data in result.masks.data:
                    mask = mask_data.cpu().numpy()
                    mask = (mask * 255).astype(np.uint8)
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    combined_mask = np.maximum(combined_mask, mask)
        
        if np.any(combined_mask):
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                largest_mask = np.zeros_like(combined_mask)
                cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                return largest_mask
    
    return None

def detect_umbrella(frame):
    """ตรวจจับร่มในเฟรมและคืนค่ากล่องรอบวัตถุ"""
    results = model_object.predict(frame, device=DEVICE)
    
    umbrella_boxes = []
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            for i, box in enumerate(boxes):
                if int(class_ids[i]) == 28:  # หมายเลขคลาสของร่ม
                    umbrella_boxes.append(box)
    
    return umbrella_boxes
