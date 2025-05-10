import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Load YOLOv3
net = cv2.dnn.readNet("../yolov3/yolov3.weights", "../yolov3/yolov3.cfg")
with open("../yolov3/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

phone_detected = False
start_time = None
end_time = None

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

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    phone_bbox = detect_phone(frame)
    hand_bboxes = detect_hands(frame)

    # Draw bounding box for detected phone
    if phone_bbox:
        x, y, w, h = phone_bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box for phone
        cv2.putText(frame, "Phone", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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

    # Display webcam feed with detected message
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
