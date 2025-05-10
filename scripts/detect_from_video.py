import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Detection state
phone_detected = False
start_time = None
end_time = None

# Detect phone using YOLOv3
def detect_phone(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    phone_bbox = None
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "cell phone":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                phone_bbox = (x, y, w, h)
    return phone_bbox

# Detect hands using MediaPipe
def detect_hands(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    hand_bboxes = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min = y_min = float('inf')
            x_max = y_max = float('-inf')
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            hand_bboxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
    return hand_bboxes

# Check overlap between phone and hand
def is_phone_in_hand(phone_bbox, hand_bboxes):
    if not phone_bbox or not hand_bboxes:
        return False
    px, py, pw, ph = phone_bbox
    phone_rect = [px, py, px + pw, py + ph]
    for hx, hy, hw, hh in hand_bboxes:
        hand_rect = [hx, hy, hx + hw, hy + hh]
        if (phone_rect[0] < hand_rect[2] and phone_rect[2] > hand_rect[0] and
            phone_rect[1] < hand_rect[3] and phone_rect[3] > hand_rect[1]):
            return True
    return False

# Open video file
video_path = "video_file_path" # Replace with your video file path
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FPS, 10)  # Try to reduce FPS

if not cap.isOpened():
    print("Error opening video file")
    exit()

target_width = 640
target_height = 480

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Process only every 3rd frame
    if frame_count % 3 != 0:
        continue

    # Resize frame for consistent processing
    frame = cv2.resize(frame, (target_width, target_height))

    # Run detections
    phone_bbox = detect_phone(frame)
    hand_bboxes = detect_hands(frame)

    if is_phone_in_hand(phone_bbox, hand_bboxes):
        if not phone_detected:
            phone_detected = True
            start_time = time.time()
            print(f"Phone usage started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        cv2.putText(frame, "Phone usage detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        if phone_detected:
            phone_detected = False
            end_time = time.time()
            duration = end_time - start_time
            start_time = None
            print(f"Phone usage ended at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))} (Duration: {duration:.2f} seconds)")
        cv2.putText(frame, "No phone usage found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2)

    time.sleep(0.03)  # Delay to reduce CPU load / frame rate
    cv2.imshow("Video", frame)

    # Press 'q' to exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
