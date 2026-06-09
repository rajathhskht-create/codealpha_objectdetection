# 🎯 Real-Time Object Detection & Tracking

A real-time **object detection and tracking** system built with Python — supports **YOLOv8** and **Faster R-CNN** as detector backends, with a self-contained **SORT tracker** (Kalman Filter + Hungarian Algorithm) for persistent multi-object tracking.

> 🎓 Built as **Task 4** of my Python internship at **CodeAlpha**

---

## 🚀 Demo

> *(Add a screen-recorded GIF or screenshot here after running the app)*
> 
> Tip: Record your webcam demo and convert to GIF using [ezgif.com](https://ezgif.com/video-to-gif)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| ⚡ YOLOv8 | Ultra-fast real-time detection (default) |
| 🧠 Faster R-CNN | High-accuracy detection alternative |
| 🎯 SORT Tracker | Kalman Filter + Hungarian Algorithm tracking |
| 🆔 Unique Track IDs | Every object gets a persistent ID across frames |
| 🌈 Color-coded Tracks | Each tracked object gets its own unique color |
| 〰️ Motion Trails | Smooth trailing lines show object movement paths |
| 📊 Live HUD | Real-time stats — model, detections, active tracks |
| 📷 Webcam Support | Live detection from any connected camera |
| 🎬 Video File Support | Run on any `.mp4`, `.avi` video file |
| ⚙️ CLI Arguments | Fully configurable via command line |
| 🖥️ Cross-platform | Windows, Mac & Linux |

---

## 🧠 How It Works

```
Video Frame
    │
    ▼
┌─────────────┐     ┌──────────────────────────────┐
│  Detector   │────▶│  YOLOv8  or  Faster R-CNN    │
│  (per frame)│     │  → bounding boxes + scores   │
└─────────────┘     └──────────────────────────────┘
    │
    ▼ detections (N × 5): [x1, y1, x2, y2, score]
┌─────────────────────────────────────────────────┐
│              SORT Tracker                        │
│                                                  │
│  1. Kalman Filter predicts next position         │
│  2. IoU matrix computed (detections × tracks)   │
│  3. Hungarian Algorithm assigns best matches    │
│  4. Unmatched detections → new tracks           │
│  5. Lost tracks removed after max_age frames    │
└─────────────────────────────────────────────────┘
    │
    ▼ tracks (M × 5): [x1, y1, x2, y2, track_id]
┌─────────────────────┐
│  Draw on Frame      │
│  • Bounding box     │
│  • Label + ID badge │
│  • Motion trail     │
│  • HUD stats        │
└─────────────────────┘
```

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **YOLOv8** (Ultralytics) — state-of-the-art real-time object detector
- **Faster R-CNN** (torchvision) — high-accuracy detector alternative
- **OpenCV** — video capture, frame drawing, display
- **NumPy** — array operations
- **SciPy** — Hungarian algorithm (`linear_sum_assignment`)
- **Kalman Filter** — built-in via `cv2.KalmanFilter` (no extra install)

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/object-detection-tracking.git
cd object-detection-tracking
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **For Faster R-CNN** (optional):
> ```bash
> pip install torch torchvision
> ```

### 4. Run the app

**Webcam (default):**
```bash
python src/object_detection.py
```

**Video file:**
```bash
python src/object_detection.py --source path/to/video.mp4
```

**Use Faster R-CNN instead of YOLO:**
```bash
python src/object_detection.py --model fasterrcnn
```

**Custom confidence threshold:**
```bash
python src/object_detection.py --conf 0.5
```

**All options combined:**
```bash
python src/object_detection.py --source video.mp4 --model yolo --conf 0.4
```

> Press **Q** to quit the window at any time.

---

## 📁 Project Structure

```
object-detection-tracking/
│
├── src/
│   └── object_detection.py      # Full pipeline — detector + tracker + visualizer
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignores model weights, videos, venv
└── README.md
```

> **Note:** YOLOv8 weights (`yolov8n.pt`) download automatically on first run (~6MB). They are gitignored.

---

## 🎛️ CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `0` | `0` = webcam, or path to video file |
| `--model` | `yolo` | `yolo` or `fasterrcnn` |
| `--conf` | `0.4` | Confidence threshold (0.0 – 1.0) |

---

## 🔬 SORT Tracker — Under the Hood

The SORT (Simple Online and Realtime Tracking) algorithm is implemented **from scratch** without any external tracking library:

| Component | Role |
|-----------|------|
| `KalmanBoxTracker` | Predicts next bounding box position using Kalman Filter with 7D state vector `[cx, cy, area, aspect, dx, dy, ds]` |
| `SORTTracker.update()` | Associates detections to existing tracks each frame |
| `iou()` | Computes Intersection-over-Union between two bounding boxes |
| `linear_sum_assignment` | Solves the optimal detection-to-track assignment (Hungarian Algorithm) |
| `id_to_color()` | Generates a unique, deterministic color per track ID |

---

## 🚀 Future Improvements

- [ ] Save output video with tracking overlay
- [ ] Object count per class in HUD
- [ ] DeepSORT (appearance-based re-identification)
- [ ] Support for IP camera streams (RTSP)
- [ ] Web dashboard using Flask/Streamlit

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

## 👤 Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [SORT: Simple, Online and Realtime Tracking](https://arxiv.org/abs/1602.00763) — Bewley et al., 2016
- **CodeAlpha** — for the internship opportunity
