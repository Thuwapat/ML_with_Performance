from ultralytics import YOLO
import numpy as np

# Load YOLO model
model_post = YOLO("yolo11n-pose.pt")
model_post.to('cuda')
model_hand = YOLO("hand_detection.pt")
model_hand.to('cuda')
model_object = YOLO("yolo11n.pt")
model_object.to('cuda')

def get_post_keypoint(frame):
    results = model_post.predict(frame)

    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()

            if keypoints.shape[1] >= 7:  # Ensure shoulders exist
                for keypoint in keypoints:
                    left_shoulder_x = keypoint[5][0] if len(keypoint) > 5 else None
                    right_shoulder_x = keypoint[6][0] if len(keypoint) > 6 else None
                    return left_shoulder_x, right_shoulder_x, keypoint  # Return keypoints too
    return None, None, []

##### Not use Yet #####
#def get_hand_keypoint(frame):
#    results = model_hand.predict(frame)
#
#    hand_boxes = []  # Store bounding boxes
#    hand_keypoints = []  # Store keypoints
#
#    left_hand, right_hand = None, None  # Default values if hands not detected
#
#    for result in results:
#        if result.boxes is not None and len(result.boxes) > 0:  # Check if hands detected
#            boxes = result.boxes.xyxy.cpu().numpy()
#            hand_boxes.extend(boxes)  # Add detected boxes
#
#        if result.keypoints is not None and len(result.keypoints) > 0:  # Check if keypoints exist
#            keypoints = result.keypoints.xy.cpu().numpy()
#            hand_keypoints.extend(keypoints)
#
#            #  Ensure at least 2 hands exist before assignment
#            if len(hand_keypoints) > 0:
#                left_hand = hand_keypoints[0][:2] if len(hand_keypoints[0]) >= 2 else None
#            if len(hand_keypoints) > 1:
#                right_hand = hand_keypoints[1][:2] if len(hand_keypoints[1]) >= 2 else None
#
#    return hand_boxes, hand_keypoints, left_hand, right_hand

# Placeholder for hand tracking 
def detect_hand(frame):
    results = model_hand.predict(frame)

    left_hand, right_hand = None, None  
    handful = False
    hand_center, hand_open = None, False  
    hands_together = False
    
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:  
            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else []
            
            if len(boxes) >= 1:
                left_hand = (int((boxes[0][0] + boxes[0][2]) / 2), int((boxes[0][1] + boxes[0][3]) / 2))

            if len(boxes) >= 2:  
                right_hand = (int((boxes[1][0] + boxes[1][2]) / 2), int((boxes[1][1] + boxes[1][3]) / 2))

            hand_center = left_hand if right_hand is None else right_hand  

            if len(keypoints) > 0:
                finger_spread = np.mean(keypoints[0][:, 1]) > boxes[0][1] + (boxes[0][3] - boxes[0][1]) * 0.5
                hand_open = finger_spread
                handful = not hand_open

            if len(boxes) > 1:  
                left_hand = (int((boxes[0][0] + boxes[0][2]) / 2), int((boxes[0][1] + boxes[0][3]) / 2))
                right_hand = (int((boxes[1][0] + boxes[1][2]) / 2), int((boxes[1][1] + boxes[1][3]) / 2))

                #  Detect if hands are close
                distance = np.sqrt((left_hand[0] - right_hand[0])**2 + (left_hand[1] - right_hand[1])**2)
                hands_together = distance < 40 

    return left_hand, right_hand, handful, hand_center, hand_open, hands_together

def detect_body(frame):
    results = model_object.predict(frame)

    body_box = None  

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:  
            boxes = result.boxes.xyxy.cpu().numpy()

            if len(boxes) > 0:
                x1, y1, x2, y2 = boxes[0]  
                body_box = (int(x1), int(y1), int(x2), int(y2))  

    return body_box