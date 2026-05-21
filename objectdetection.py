"""
TASK 4: Object Detection and Tracking
--------------------------------------
Requirements:
    pip install ultralytics opencv-python numpy scipy

Usage:
    # Webcam
    python object_detection_tracking.py

    # Video file
    python object_detection_tracking.py --source path/to/video.mp4

    # Use Faster R-CNN instead of YOLO
    python object_detection_tracking.py --model fasterrcnn
"""

import cv2
import numpy as np
import argparse
from collections import deque

# ──────────────────────────────────────────────────────────────────────────────
# SORT TRACKER  (minimal self-contained implementation — no extra install)
# ──────────────────────────────────────────────────────────────────────────────

from scipy.optimize import linear_sum_assignment


def iou(bb_test, bb_gt):
    """Intersection-over-Union between two bounding boxes [x1,y1,x2,y2]."""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt   = (bb_gt[2]   - bb_gt[0])   * (bb_gt[3]   - bb_gt[1])
    union = area_test + area_gt - inter
    return inter / union if union > 0 else 0.0


class KalmanBoxTracker:
    """Tracks a single object bounding box with a Kalman filter."""
    count = 0

    def __init__(self, bbox):
        # State: [x, y, s, r, dx, dy, ds]  (center-x, center-y, area, aspect, velocities)
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.measurementMatrix = np.array(
            [[1,0,0,0,0,0,0],
             [0,1,0,0,0,0,0],
             [0,0,1,0,0,0,0],
             [0,0,0,1,0,0,0]], dtype=np.float32)
        self.kf.transitionMatrix = np.array(
            [[1,0,0,0,1,0,0],
             [0,1,0,0,0,1,0],
             [0,0,1,0,0,0,1],
             [0,0,0,1,0,0,0],
             [0,0,0,0,1,0,0],
             [0,0,0,0,0,1,0],
             [0,0,0,0,0,0,1]], dtype=np.float32)
        self.kf.processNoiseCov     *= 0.01
        self.kf.measurementNoiseCov *= 10
        self.kf.errorCovPost        *= 1000
        self.kf.statePost = np.vstack([self._bbox_to_z(bbox), np.zeros((3, 1), dtype=np.float32)])

        KalmanBoxTracker.count += 1
        self.id           = KalmanBoxTracker.count
        self.hits         = 0
        self.no_losses    = 0
        self.age          = 0
        self.history      = deque(maxlen=30)

    @staticmethod
    def _bbox_to_z(bbox):
        """[x1,y1,x2,y2] → [cx, cy, area, aspect]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s  = w * h
        r  = w / float(h) if h else 1.0
        return np.array([[cx], [cy], [s], [r]], dtype=np.float32)

    @staticmethod
    def _z_to_bbox(z, score=0):
        """[cx, cy, area, aspect] → [x1, y1, x2, y2, score]"""
        w = np.sqrt(abs((z[2] * z[3]).item()))
        h = z[2].item() / w if w else 0
        return [
            int(z[0].item() - w / 2), int(z[1].item() - h / 2),
            int(z[0].item() + w / 2), int(z[1].item() + h / 2),
            score
        ]

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.no_losses += 1
        pred = self._z_to_bbox(self.kf.statePost)
        self.history.append(pred[:4])
        return pred

    def update(self, bbox):
        self.no_losses = 0
        self.hits += 1
        z = self._bbox_to_z(bbox)
        self.kf.correct(z)

    def get_state(self):
        return self._z_to_bbox(self.kf.statePost)


class SORTTracker:
    """
    Simple Online and Realtime Tracking (SORT).
    Pairs detections to existing tracks using IoU + Hungarian algorithm.
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers      = []
        self.frame_count   = 0

    def update(self, detections):
        """
        detections : np.ndarray  shape (N, 5)  → [x1, y1, x2, y2, score]
        returns     : np.ndarray  shape (M, 5)  → [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1

        # Predict positions for existing trackers
        predicted = []
        for t in self.trackers:
            predicted.append(t.predict()[:4])

        # ── Associate detections to trackers ──────────────────────────────
        matched, unmatched_det, unmatched_trk = self._associate(
            detections, predicted
        )

        # Update matched trackers
        for d_idx, t_idx in matched:
            self.trackers[t_idx].update(detections[d_idx, :4])

        # Create new trackers for unmatched detections
        for d_idx in unmatched_det:
            self.trackers.append(KalmanBoxTracker(detections[d_idx, :4]))

        # Remove dead trackers
        alive = []
        results = []
        for t in self.trackers:
            if t.no_losses <= self.max_age:
                if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                    state = t.get_state()
                    results.append([*state[:4], t.id])
                alive.append(t)
        self.trackers = alive

        return np.array(results, dtype=np.int32) if results else np.empty((0, 5), dtype=np.int32)

    def _associate(self, detections, predictions):
        if len(predictions) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(predictions)))

        iou_matrix = np.zeros((len(detections), len(predictions)))
        for d, det in enumerate(detections):
            for p, pred in enumerate(predictions):
                iou_matrix[d, p] = iou(det[:4], pred)

        d_indices, p_indices = linear_sum_assignment(-iou_matrix)

        matched, unmatched_det, unmatched_trk = [], [], []
        for d in range(len(detections)):
            if d not in d_indices:
                unmatched_det.append(d)
        for p in range(len(predictions)):
            if p not in p_indices:
                unmatched_trk.append(p)
        for d, p in zip(d_indices, p_indices):
            if iou_matrix[d, p] < self.iou_threshold:
                unmatched_det.append(d)
                unmatched_trk.append(p)
            else:
                matched.append((d, p))

        return matched, unmatched_det, unmatched_trk


# ──────────────────────────────────────────────────────────────────────────────
# DETECTOR WRAPPERS
# ──────────────────────────────────────────────────────────────────────────────

class YOLODetector:
    """YOLOv8 detector via Ultralytics."""

    def __init__(self, model_size="n", conf=0.4):
        from ultralytics import YOLO
        model_name = f"yolov8{model_size}.pt"
        print(f"[YOLO] Loading {model_name} …")
        self.model  = YOLO(model_name)
        self.conf   = conf
        self.names  = self.model.names

    def detect(self, frame):
        """Returns (boxes_xyxy, scores, class_ids)."""
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        boxes   = results.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 4)), [], []
        xyxy   = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        cls    = boxes.cls.cpu().numpy().astype(int)
        return xyxy, scores, cls


class FasterRCNNDetector:
    """Faster R-CNN detector via torchvision."""

    def __init__(self, conf=0.5):
        import torch
        import torchvision
        print("[FasterRCNN] Loading model …")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        ).to(self.device).eval()
        self.conf  = conf
        self.names = {   # COCO class names
            1:"person", 2:"bicycle", 3:"car", 4:"motorcycle", 5:"airplane",
            6:"bus", 7:"train", 8:"truck", 9:"boat", 10:"traffic light",
            11:"fire hydrant", 13:"stop sign", 14:"parking meter", 15:"bench",
            16:"bird", 17:"cat", 18:"dog", 19:"horse", 20:"sheep", 21:"cow",
            22:"elephant", 23:"bear", 24:"zebra", 25:"giraffe", 27:"backpack",
            28:"umbrella", 31:"handbag", 32:"tie", 33:"suitcase", 34:"frisbee",
            35:"skis", 36:"snowboard", 37:"sports ball", 38:"kite",
            39:"baseball bat", 40:"baseball glove", 41:"skateboard",
            42:"surfboard", 43:"tennis racket", 44:"bottle", 46:"wine glass",
            47:"cup", 48:"fork", 49:"knife", 50:"spoon", 51:"bowl",
            52:"banana", 53:"apple", 54:"sandwich", 55:"orange", 56:"broccoli",
            57:"carrot", 58:"hot dog", 59:"pizza", 60:"donut", 61:"cake",
            62:"chair", 63:"couch", 64:"potted plant", 65:"bed",
            67:"dining table", 70:"toilet", 72:"tv", 73:"laptop",
            74:"mouse", 75:"remote", 76:"keyboard", 77:"cell phone",
            78:"microwave", 79:"oven", 80:"toaster", 81:"sink",
            82:"refrigerator", 84:"book", 85:"clock", 86:"vase",
            87:"scissors", 88:"teddy bear", 89:"hair drier", 90:"toothbrush"
        }

    def detect(self, frame):
        import torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy((rgb / 255.0).copy()).permute(2, 0, 1).float()
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(tensor)[0]

        keep   = preds["scores"] >= self.conf
        boxes  = preds["boxes"][keep].cpu().numpy()
        scores = preds["scores"][keep].cpu().numpy()
        cls    = preds["labels"][keep].cpu().numpy().astype(int)
        return boxes, scores, cls


# ──────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (unique colour per track ID)
# ──────────────────────────────────────────────────────────────────────────────

def id_to_color(track_id):
    np.random.seed(track_id * 31 + 17)
    return tuple(int(c) for c in np.random.randint(50, 230, 3))


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run(source=0, model_type="yolo", conf=0.4):
    # ── Detector ──────────────────────────────────────────────────────────────
    if model_type == "yolo":
        detector = YOLODetector(model_size="n", conf=conf)
    elif model_type == "fasterrcnn":
        detector = FasterRCNNDetector(conf=conf)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    # ── Tracker ───────────────────────────────────────────────────────────────
    tracker = SORTTracker(max_age=30, min_hits=3, iou_threshold=0.3)

    # ── Video capture ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    # Keep short trails for each track
    trails = {}          # track_id → deque of center points

    print("\n▶  Running — press  Q  to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # ── 1. Detect ─────────────────────────────────────────────────────────
        boxes, scores, class_ids = detector.detect(frame)

        # Pack into (N,5) array for SORT
        if len(boxes) > 0:
            dets = np.hstack([boxes, scores.reshape(-1, 1)])
        else:
            dets = np.empty((0, 5))

        # ── 2. Track ──────────────────────────────────────────────────────────
        tracks = tracker.update(dets)   # → [x1, y1, x2, y2, id]

        # ── 3. Draw ───────────────────────────────────────────────────────────
        for trk in tracks:
            x1, y1, x2, y2, tid = trk
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            color = id_to_color(tid)

            # Trail
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            trails.setdefault(tid, deque(maxlen=20)).append((cx, cy))
            pts = list(trails[tid])
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, pts[i-1], pts[i], trail_color, 2)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Try to find the best matching class label for this track
            label = "object"
            if len(boxes) > 0:
                best_iou, best_cls = 0, None
                for j, box in enumerate(boxes):
                    score = iou([x1,y1,x2,y2], box)
                    if score > best_iou:
                        best_iou = score
                        best_cls = class_ids[j]
                if best_cls is not None and hasattr(detector, "names"):
                    label = detector.names.get(int(best_cls), f"cls{best_cls}")

            # Label badge
            tag = f"{label} | ID:{tid}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ── 4. HUD ────────────────────────────────────────────────────────────
        hud = (f"Model: {model_type.upper()}  |  "
               f"Detections: {len(boxes)}  |  "
               f"Tracks: {len(tracks)}")
        (hud_w, hud_h), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (0, 0), (hud_w + 12, 28), (0, 0, 0), -1)
        cv2.putText(frame, hud, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Object Detection & Tracking  [Q to quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection & Tracking")
    parser.add_argument(
        "--source", default=0,
        help="Video source: 0 for webcam, or path to a video file"
    )
    parser.add_argument(
        "--model", default="yolo", choices=["yolo", "fasterrcnn"],
        help="Detector backend (default: yolo)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.4,
        help="Detection confidence threshold (default: 0.4)"
    )
    args = parser.parse_args()

    # Allow numeric string source ("0", "1", …) as integers
    source = int(args.source) if str(args.source).isdigit() else args.source

    run(source=source, model_type=args.model, conf=args.conf)