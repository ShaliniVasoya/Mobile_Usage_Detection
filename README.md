
# 📱 Mobile Phone Usage Detection with YOLOv3 and MediaPipe

This project uses YOLOv3 and MediaPipe to detect when a person is using a mobile phone in real-time through a webcam or videofeed. It highlights the phone and hands on the video feed and logs when phone usage starts and ends.

---

# 📁 Project Structure
Copy
Edit
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


---

## 🔗 Download YOLOv3 Model Files

Place the following files in the `yolov3/` directory:

- [`yolov3.cfg`](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
- [`yolov3.weights`](https://pjreddie.com/media/files/yolov3.weights)
- [`coco.names`](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

---

## ▶️ How to Run

1. Setup yolov3 model with this project and save model in yolov3 folder.
2. Run the detection script:

▶️ How to Run
🔴 1. Webcam Mode
To detect phone usage from your webcam:

bash
Copy
Edit
cd scripts
python detect_from_webcam.py
🎥 2. Video File Mode
To detect phone usage in a saved video file:

bash
Copy
Edit
cd scripts
python detect_from_video.py 
📝 Replace path/to/your/video.mp4 with the actual path to your video file.

📌 What It Does
Detects mobile phones using YOLOv3.

Detects hands using MediaPipe.

Checks if a phone is being held in a hand.

Logs the start and end time of phone usage.

Displays visual feedback on the webcam feed.

❌ Exit Instructions
Press q to quit the webcam or video window anytime.

vbnet
Copy
Edit
