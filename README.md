# 📱 Mobile Phone Usage Detection with YOLOv3 and MediaPipe

This project uses YOLOv3 and MediaPipe to detect when a person is using a mobile phone in real-time through a webcam or video feed. It highlights the phone and hands on the video feed and logs when phone usage starts and ends.

## 📁 Project Structure

```
project-root/
├── yolov3/
│   ├── yolov3.cfg
│   ├── yolov3.weights
│   ├── coco.names
├── scripts/
│   ├── detect_from_video.py
│   ├── detect_from_webcam.py
├── requirements.txt
├── README.md
```

## 🔗 Download YOLOv3 Model Files

Place the following files in the `yolov3/` directory:

- [`yolov3.cfg`](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
- [`yolov3.weights`](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights)
- [`coco.names`](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

## 📋 Requirements

```
pip install -r requirements.txt
```

## ▶️ How to Run

### 🔴 1. Webcam Mode
To detect phone usage from your webcam:

```bash
cd scripts
python detect_from_webcam.py
```

### 🎥 2. Video File Mode
To detect phone usage in a saved video file:

```bash
cd scripts
python detect_from_video.py --input path/to/your/video.mp4
```

*Replace `path/to/your/video.mp4` with the actual path to your video file.*

## 📌 What It Does

- Detects mobile phones using YOLOv3
- Detects hands using MediaPipe
- Checks if a phone is being held in a hand
- Logs the start and end time of phone usage
- Displays visual feedback on the webcam feed

## ❌ Exit Instructions

Press `q` to quit the webcam or video window anytime.

## 📊 Output

The system will display:
- Bounding boxes around detected phones
- Hand landmarks when hands are detected
- On-screen status of phone usage
- Terminal logs of phone usage events

## 🛠️ Troubleshooting

- Make sure all YOLOv3 files are correctly placed in the yolov3 directory
- Check camera permissions if webcam mode doesn't work
- For video processing issues, ensure your video format is supported by OpenCV