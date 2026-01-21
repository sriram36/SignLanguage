import base64
import io
import math
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from keras.models import load_model
import mediapipe as mp
from PIL import Image

# App and model setup
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cnn8grps_rad1_model.h5"
CANVAS = 400
OFFSET = 29
CONF_THRESHOLD = 0.6
SMOOTH_WINDOW = 5

app = Flask(__name__, static_folder="static", static_url_path="")


# MediaPipe hands (static mode for single images)
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def load_sign_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return load_model(MODEL_PATH)


model = load_sign_model()
recent_chars: deque[str] = deque(maxlen=SMOOTH_WINDOW)


def distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))


def draw_skeleton(img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List[Tuple[int, int]]]]:
    """Run hand detection and render a 400x400 skeleton image."""
    img_bgr = cv2.flip(img_bgr, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    results = mp_hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None, None

    lm = results.multi_hand_landmarks[0]
    pts_px: List[Tuple[float, float]] = []
    min_x = min_y = 1e9
    max_x = max_y = -1e9
    for p in lm.landmark:
        x = p.x * w
        y = p.y * h
        pts_px.append((x, y))
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    box_w = max(10.0, max_x - min_x)
    box_h = max(10.0, max_y - min_y)
    scale = min((CANVAS - 2 * OFFSET) / box_w, (CANVAS - 2 * OFFSET) / box_h)

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0

    pts_canvas: List[Tuple[int, int]] = []
    for x, y in pts_px:
        nx = int((x - cx) * scale + CANVAS / 2)
        ny = int((y - cy) * scale + CANVAS / 2)
        pts_canvas.append((nx, ny))

    canvas = np.ones((CANVAS, CANVAS, 3), dtype=np.uint8) * 255

    # Draw finger chains
    chains = [
        range(0, 4),  # thumb 0-1-2-3-4
        range(5, 8),  # index 5-6-7-8
        range(9, 12),  # middle 9-10-11-12
        range(13, 16),  # ring
        range(17, 20),  # pinky
    ]
    for chain in chains:
        for i in chain:
            cv2.line(canvas, pts_canvas[i], pts_canvas[i + 1], (0, 255, 0), 3)

    # Palm links
    palm_links = [(5, 9), (9, 13), (13, 17), (0, 5), (0, 17)]
    for a, b in palm_links:
        cv2.line(canvas, pts_canvas[a], pts_canvas[b], (0, 255, 0), 3)

    for pt in pts_canvas:
        cv2.circle(canvas, pt, 2, (0, 0, 255), 1)

    return canvas, pts_canvas


def predict_character(skel_img: np.ndarray, pts: List[Tuple[int, int]]) -> Optional[str]:
    white = skel_img.reshape(1, CANVAS, CANVAS, 3)
    prob = np.array(model.predict(white)[0], dtype="float32")
    if float(np.max(prob)) < CONF_THRESHOLD:
        return None
    ch1 = int(np.argmax(prob, axis=0))
    prob[ch1] = 0
    ch2 = int(np.argmax(prob, axis=0))
    prob[ch2] = 0
    ch3 = int(np.argmax(prob, axis=0))
    prob[ch3] = 0

    def cdist(i, j):
        return distance(pts[i], pts[j])

    pl = [ch1, ch2]

    l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
         [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
         [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 0

    l = [[2, 2], [2, 1]]
    if pl in l:
        if pts[5][0] < pts[4][0]:
            ch1 = 0

    l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and pts[5][0] > pts[4][0]:
            ch1 = 2

    l = [[6, 0], [6, 6], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if cdist(8, 16) < 52:
            ch1 = 2

    l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
            ch1 = 3

    l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 3

    l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[2][1] + 15 < pts[16][1]:
            ch1 = 3

    l = [[6, 4], [6, 1], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if cdist(4, 11) > 55:
            ch1 = 4

    l = [[1, 4], [1, 6], [1, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (cdist(4, 11) > 50) and (
            pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]
        ):
            ch1 = 4

    l = [[3, 6], [3, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] < pts[0][0]:
            ch1 = 4

    l = [[2, 2], [2, 5], [2, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[1][0] < pts[12][0]:
            ch1 = 4

    l = [[3, 6], [3, 5], [3, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
        ) and pts[4][1] > pts[10][1]:
            ch1 = 5

    l = [[3, 2], [3, 1], [3, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][1] + 17 > pts[8][1] and pts[4][1] + 17 > pts[12][1] and pts[4][1] + 17 > pts[16][1] and pts[4][1] + 17 > pts[20][1]:
            ch1 = 5

    l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 5

    l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
            ch1 = 5

    l = [[5, 7], [5, 2], [5, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[3][0] < pts[0][0]:
            ch1 = 7

    l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] < pts[8][1]:
            ch1 = 7

    l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] > pts[20][1]:
            ch1 = 7

    l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] > pts[16][0]:
            ch1 = 6

    l = [[7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]:
            ch1 = 6

    l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if cdist(8, 16) > 50:
            ch1 = 6

    l = [[4, 6], [4, 2], [4, 1], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if cdist(4, 11) < 60:
            ch1 = 6

    l = [[1, 4], [1, 6], [1, 0], [1, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 > 0:
            ch1 = 6

    l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
         [6, 3], [6, 4], [7, 5], [7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
         [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if (
            (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1])
            and (pts[2][0] < pts[0][0])
            and pts[4][1] > pts[14][1]
        ):
            ch1 = 1

    l = [[4, 1], [4, 2], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (cdist(4, 11) < 50) and (
            pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]
        ):
            ch1 = 1

    l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if (
            (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1])
            and (pts[2][0] < pts[0][0])
            and pts[14][1] < pts[4][1]
        ):
            ch1 = 1

    l = [[6, 6], [6, 4], [6, 1], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 < 0:
            ch1 = 1

    l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[4][0] < pts[5][0] + 15) and (
            pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]
        ):
            ch1 = 7

    l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if (
            (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1])
            and pts[4][1] > pts[14][1]
        ):
            ch1 = 1

    l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
    pl = [ch1, ch2]
    fg = 13
    if pl in l:
        if not (pts[0][0] + fg < pts[8][0] and pts[0][0] + fg < pts[12][0] and pts[0][0] + fg < pts[16][0] and pts[0][0] + fg < pts[20][0]) and not (
            pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]
        ) and cdist(4, 11) < 50:
            ch1 = 1

    l = [[5, 0], [5, 5], [0, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1]:
            ch1 = 1

    # map to final letters
    if ch1 == 0:
        ch1 = "S"
        if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
            ch1 = "A"
        if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]:
            ch1 = "T"
        if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
            ch1 = "E"
        if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]:
            ch1 = "M"
        if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
            ch1 = "N"

    if ch1 == 2:
        if cdist(12, 4) > 42:
            ch1 = "C"
        else:
            ch1 = "O"

    if ch1 == 3:
        if cdist(8, 12) > 72:
            ch1 = "G"
        else:
            ch1 = "H"

    if ch1 == 7:
        if cdist(8, 4) > 42:
            ch1 = "Y"
        else:
            ch1 = "J"

    if ch1 == 4:
        ch1 = "L"

    if ch1 == 6:
        ch1 = "X"

    if ch1 == 5:
        if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
            if pts[8][1] < pts[5][1]:
                ch1 = "Z"
            else:
                ch1 = "Q"
        else:
            ch1 = "P"

    if ch1 == 1:
        if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
            ch1 = "B"
        if pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]:
            ch1 = "D"
        if pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
            ch1 = "F"
        if pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]:
            ch1 = "I"
        if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]:
            ch1 = "W"
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
        ) and pts[4][1] < pts[9][1]:
            ch1 = "K"
        if (cdist(8, 12) - cdist(6, 10)) < 8 and (
            pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]
        ):
            ch1 = "U"
        if (cdist(8, 12) - cdist(6, 10)) >= 8 and (
            pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]
        ) and (pts[4][1] > pts[9][1]):
            ch1 = "V"
        if pts[8][0] > pts[12][0] and (
            pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]
        ):
            ch1 = "R"

    if isinstance(ch1, str):
        return ch1
    return None


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify({"error": "missing image"}), 400
        payload = data["image"]
        if "," in payload:
            payload = payload.split(",", 1)[1]
        decoded = base64.b64decode(payload)
        img = Image.open(io.BytesIO(decoded)).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        skeleton, pts = draw_skeleton(frame)
        if skeleton is None or pts is None:
            return jsonify({"error": "hand not detected"}), 422
        char = predict_character(skeleton, pts)
        if not char:
            return jsonify({"error": "could not classify"}), 422
        recent_chars.append(char)
        if recent_chars:
            counts = {}
            for c in recent_chars:
                counts[c] = counts.get(c, 0) + 1
            smooth_char = max(counts, key=counts.get)
        else:
            smooth_char = char
        return jsonify({"char": smooth_char})
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
