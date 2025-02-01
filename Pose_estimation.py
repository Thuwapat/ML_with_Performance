from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n-pose.pt")

def get_shoulder_x(frame):
    results = model(frame)

    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()

            if keypoints.shape[1] >= 7:  # Ensure shoulders exist
                for keypoint in keypoints:
                    left_shoulder_x = keypoint[5][0] if len(keypoint) > 5 else None
                    right_shoulder_x = keypoint[6][0] if len(keypoint) > 6 else None
                    return left_shoulder_x, right_shoulder_x, keypoint  # Return keypoints too
    return None, None, []
