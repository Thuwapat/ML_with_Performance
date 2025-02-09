from ultralytics import YOLO
import numpy as np

# Load YOLO model
model_post = YOLO("yolo11n-pose.pt")
model_post.to('cuda')
model_hand = YOLO("hand_detection.pt")
model_hand.to('cuda')

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

    hand_center, hand_open = None, False  # Default values if no hand is detected

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:  # Ensure detection exists
            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else []

            # Use first detected hand (for now)
            x1, y1, x2, y2 = boxes[0][:4]
            hand_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))  # Center of bounding box

            # Check if fingers are spread (i.e., open palm)
            if len(keypoints) > 0:
                finger_spread = np.mean(keypoints[0][:, 1]) > y1 + (y2 - y1) * 0.5  # Simple spread detection
                hand_open = finger_spread  # If fingers are high above palm, assume open

    return hand_center, hand_open  # Return position + open/closed state