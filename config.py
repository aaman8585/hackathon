"""Configuration for the Billboard Attention Tracker (MVP).

Defaults are the v2 screenshot-angle values from the product spec.
Tune the 0.10 gaze thresholds only on the 30 hand-labeled people
(15 look, 15 no-look) from the demo clip. Do not train a custom
gaze model.
"""

# Input: "stream" for the product, "file" for the demo.
INPUT = {
    "source": "file",
    "stream_url": "https://www.youtube.com/watch?v=AdUw5RdyZxI",
    "file_path": "data/demo_clip.mp4",
    "capture_seconds": 180,
}

# Billboard ROI in normalized frame coords [0, 1]: (x0, y0, x1, y1).
# Default: Welcome to Times Square screen, centre-left (v2 angle).
BILLBOARD_ROI = (0.24, 0.11, 0.44, 0.38)
# Fallbacks, one-line swap if the Welcome screen is static in the
# demo clip (spec section 5B). Document the choice with a screenshot.
ILLINOIS_ROI = (0.44, 0.19, 0.57, 0.42)
BROADWAY_1540_ROI = (0.78, 0.19, 0.97, 0.41)

# Gaze polygon in normalized frame coords, trapezoid over the
# foreground plaza. Anyone outside the polygon is not counted.
GAZE_POLYGON = [(0.05, 0.97), (0.65, 0.97), (0.55, 0.65), (0.10, 0.65)]

# Red-steps exclusion: people seated here are stationary, not walk-bys.
# Normalized (x0, y0, x1, y1); a bbox centre inside it is not counted.
RED_STEPS_RECT = (0.0, 0.88, 0.20, 1.0)

# Sampling.
SAMPLE_FPS = 2
ROI_SNAPSHOT_INTERVAL_S = 0.5

# Ad segmentation. Distances are L1 distances on normalized
# 8x8x8 colour histograms plus mean-colour distance (see pipeline.py).
# "Tuned on demo clip hand log" in the spec: these are starting
# values, verified on synthetic solid-colour segments in tests.
AD_CUT_THRESHOLD = 0.30
AD_CLUSTER_THRESHOLD = 0.30
# Ignore flicker / fades shorter than this; segments shorter than
# MIN_SEGMENT_FRAMES are merged into their neighbour.
MIN_SEGMENT_SECONDS = 1.0

# Gaze gates (spec FR5). YOLO-pose keypoints: 0=nose, 3/4=ears,
# 5/6=shoulders.
ORIENTATION_TOLERANCE_DEG = 45
FACING_OFFSET = 0.10
HEAD_UP_OFFSET = 0.10
NOSE_CONF_MIN = 0.3
BACK_VIEW_SHOULDER_RATIO = 0.30
HOLD_FRAMES = 3  # ~1.0s at 2fps; tech doc uses 3 samples (~1.5s)
HOLD_DURATION_S = 1.0
MIN_PERSON_HEIGHT_PX = 40  # below this: walk-by only, never a look

# Tracking fallback (centroid + IoU, used when the detector
# returns boxes without track ids).
TRACK_IOU_THRESHOLD = 0.3

# Outputs.
FRAMES_DIR = "frames"
DATA_DIR = "data"
DEBUG_FRAMES_DIR = "debug_frames"
THUMBNAILS_DIR = "data/thumbnails"

# S3 layout for the required upload (see storage.py).
S3_RESULTS_PREFIX = "billboard-demo/results"
S3_CLIPS_PREFIX = "billboard-demo/clips"

# Raw clips retained <= 7 days for debugging, then deleted.
# Only aggregate events + blurred frames are retained long-term.
RAW_RETENTION_DAYS = 7
