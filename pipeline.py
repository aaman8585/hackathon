"""Pipeline: frames -> passes.csv, ad_segments.csv, ad_summary.csv.

Batch-first, capture-then-analyse. The same code runs on a
stream capture and on the pre-captured demo clip.

Pure logic (gates, segmentation, tracking fallback, aggregation)
uses only the standard library and is unit-tested without any
vision dependency. OpenCV and Ultralytics are imported lazily
inside the frame-processing functions, so importing this module
never requires them. Detection / pose estimators are injectable,
so tests can run the full pipeline on synthetic frames with
stub estimators and a stub upload function.
"""

import csv
import json
import os
import statistics

import config


# ---------------------------------------------------------------- gates

def point_in_polygon(x, y, polygon):
    """Ray-casting point-in-polygon test. Polygon is [(x, y), ...]."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_at_y:
                inside = not inside
    return inside


def evaluate_gates(keypoints, bbox_h):
    """Apply gaze gates G1 (facing) and G2 (head up) for one person.

    ``keypoints`` is a list of (x, y, conf) in pixel coords with
    YOLO-pose indices 0=nose, 5/6=shoulders, or ``None`` when no
    pose was matched. Returns (g1, g2, scored). ``scored`` is
    False when the person is too small to judge (bbox < 40px):
    counted as a walk-by only, never as a look.
    """
    scored = bbox_h >= config.MIN_PERSON_HEIGHT_PX
    if not scored or not keypoints or len(keypoints) < 7:
        return False, False, scored
    nose_x, nose_y, nose_conf = keypoints[0]
    left_x, left_y, _ = keypoints[5]
    right_x, right_y, _ = keypoints[6]
    shoulder_mid_x = (left_x + right_x) / 2.0
    shoulder_mid_y = (left_y + right_y) / 2.0
    shoulder_w = abs(right_x - left_x)
    if shoulder_w <= 0:
        return False, False, scored
    facing = (nose_x - shoulder_mid_x) < -config.FACING_OFFSET * shoulder_w
    facing = facing or (
        nose_conf < config.NOSE_CONF_MIN
        and shoulder_w / bbox_h > config.BACK_VIEW_SHOULDER_RATIO
    )
    head_up = (
        (shoulder_mid_y - nose_y) > config.HEAD_UP_OFFSET * shoulder_w
        and nose_conf > config.NOSE_CONF_MIN
    )
    return facing, head_up, scored


def evaluate_pass(observations, fps):
    """Evaluate G3 (holds) for one pass.

    ``observations`` is a list of (g1, g2, scored) in frame order.
    Returns (is_look, dwell_look_s). Dwell is the total time G1+G2
    held; a look needs HOLD_FRAMES consecutive qualifying frames.
    """
    qualifying = [bool(g1 and g2 and scored) for g1, g2, scored in observations]
    dwell_look_s = sum(qualifying) / fps if fps else 0.0
    run = 0
    is_look = False
    for ok in qualifying:
        run = run + 1 if ok else 0
        if run >= config.HOLD_FRAMES:
            is_look = True
    return is_look, dwell_look_s


# ------------------------------------------------------- ad segmentation

def fingerprint_distance(fp1, fp2):
    """Distance between ROI fingerprints (hist, mean colour).

    ``fp`` is (hist, mean) where hist is a normalized 8x8x8
    histogram flattened to 512 values summing to 1, and mean is
    a (B, G, R) mean colour. Result is in [0, 1]: 0 for
    identical ROIs, ~0.8 for solid red vs. solid blue.
    """
    hist1, mean1 = fp1
    hist2, mean2 = fp2
    hist_l1 = sum(abs(a - b) for a, b in zip(hist1, hist2)) / 2.0
    mean_dist = sum(abs(a - b) for a, b in zip(mean1, mean2)) / (3 * 255.0)
    return 0.5 * hist_l1 + 0.5 * mean_dist


def segment_fingerprints(fingerprints, frame_times, fps, cut_threshold=None,
                         min_segment_seconds=None):
    """Split per-frame fingerprints into stable ad segments.

    A new segment starts when the fingerprint distance from the
    current segment's first fingerprint exceeds ``cut_threshold``.
    Segments shorter than ``min_segment_seconds`` (flicker /
    fades) are merged into the previous segment, or the next one
    for a leading short segment. Returns segment dicts with
    start/end indices, times, duration, and prototype fingerprint.
    """
    cut_threshold = config.AD_CUT_THRESHOLD if cut_threshold is None else cut_threshold
    min_segment_seconds = (
        config.MIN_SEGMENT_SECONDS if min_segment_seconds is None
        else min_segment_seconds
    )
    if not fingerprints:
        return []
    min_frames = max(1, int(round(min_segment_seconds * fps)))
    raw = []
    start = 0
    for i in range(1, len(fingerprints)):
        if fingerprint_distance(fingerprints[start], fingerprints[i]) > cut_threshold:
            raw.append([start, i])
            start = i
    raw.append([start, len(fingerprints)])
    # Each segment is [start_idx, end_idx, prototype_idx]. The
    # prototype stays with the segment it was measured from when
    # a short segment is merged, so a leading flicker adopts the
    # next segment's fingerprint instead of keeping its own.
    segments = []
    for start_idx, end_idx in raw:
        if segments and (end_idx - start_idx) < min_frames:
            segments[-1][1] = end_idx
        else:
            segments.append([start_idx, end_idx, start_idx])
    if len(segments) > 1 and (segments[0][1] - segments[0][0]) < min_frames:
        segments[1][0] = segments[0][0]
        segments.pop(0)
    # Coalesce neighbours that are the same creative: removing a
    # flicker can leave two same-colour segments adjacent.
    merged = []
    for start_idx, end_idx, prototype_idx in segments:
        if merged and fingerprint_distance(
                fingerprints[merged[-1][2]],
                fingerprints[prototype_idx]) <= cut_threshold:
            merged[-1][1] = end_idx
        else:
            merged.append([start_idx, end_idx, prototype_idx])
    out = []
    for start_idx, end_idx, prototype_idx in merged:
        started_at = frame_times[start_idx]
        duration_s = (end_idx - start_idx) / fps
        out.append({
            "start_idx": start_idx,
            "end_idx": end_idx,
            "started_at": started_at,
            "ended_at": started_at + duration_s,
            "duration_s": duration_s,
            "prototype": fingerprints[prototype_idx],
        })
    return out


def assign_ad_ids(segments, cluster_threshold=None):
    """Cluster segments by visual similarity into ad_ids.

    Same creative recurring = same ad_id (ad_A, ad_B, ...).
    Identity is the visual cluster only; no brand or text is
    inferred. Mutates segments with ad_id, fingerprint_id and
    confidence, and returns the segments.
    """
    cluster_threshold = (
        config.AD_CLUSTER_THRESHOLD if cluster_threshold is None
        else cluster_threshold
    )
    prototypes = []
    for segment in segments:
        matched = None
        best_distance = None
        for idx, prototype in enumerate(prototypes):
            distance = fingerprint_distance(segment["prototype"], prototype)
            if distance <= cluster_threshold and (
                best_distance is None or distance < best_distance
            ):
                matched, best_distance = idx, distance
        if matched is None:
            prototypes.append(segment["prototype"])
            matched, best_distance = len(prototypes) - 1, 0.0
        letter = chr(ord("A") + matched)
        segment["ad_id"] = f"ad_{letter}"
        segment["fingerprint_id"] = f"fp_{letter}"
        segment["confidence"] = max(0.0, 1.0 - best_distance)
    return segments


# ------------------------------------------------------------- tracking

def box_iou(box1, box2):
    """Intersection over union for (x1, y1, x2, y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def assign_track_ids(prev_tracks, boxes, next_track_id, iou_threshold=None):
    """Fallback tracker: centroid + IoU matching in ~30 lines.

    ``prev_tracks`` maps track_id -> box from the previous frame.
    Returns (assignments, next_track_id) where assignments is a
    list of (box_index, track_id) aligned with ``boxes``.
    """
    iou_threshold = (
        config.TRACK_IOU_THRESHOLD if iou_threshold is None else iou_threshold
    )
    pairs = []
    for box_idx, box in enumerate(boxes):
        for track_id, prev_box in prev_tracks.items():
            pairs.append((box_iou(box, prev_box), box_idx, track_id))
    pairs.sort(reverse=True)
    used_boxes, used_tracks, assignments = set(), set(), {}
    for iou, box_idx, track_id in pairs:
        if iou <= iou_threshold or box_idx in used_boxes or track_id in used_tracks:
            continue
        assignments[box_idx] = track_id
        used_boxes.add(box_idx)
        used_tracks.add(track_id)
    result = []
    for box_idx in range(len(boxes)):
        if box_idx not in assignments:
            assignments[box_idx] = next_track_id
            next_track_id += 1
        result.append((box_idx, assignments[box_idx]))
    return result, next_track_id


# ---------------------------------------------------------- aggregation

def majority_ad(frame_ad_ids):
    """Ad active for the majority of a pass; ties go to the earliest."""
    counts = {}
    first_seen = {}
    for order, ad_id in enumerate(frame_ad_ids):
        counts[ad_id] = counts.get(ad_id, 0) + 1
        first_seen.setdefault(ad_id, order)
    return min(counts, key=lambda ad: (-counts[ad], first_seen[ad]))


def aggregate_passes(passes, segments):
    """Aggregate passes per ad_id. Always show walk_bys with look_rate."""
    ad_seconds = {}
    for segment in segments:
        ad_seconds[segment["ad_id"]] = (
            ad_seconds.get(segment["ad_id"], 0.0) + segment["duration_s"]
        )
    by_ad = {}
    for ad_id in ad_seconds:
        by_ad[ad_id] = {"walk_bys": 0, "looks": 0, "dwell": []}
    for p in passes:
        row = by_ad.setdefault(
            p["ad_id"], {"walk_bys": 0, "looks": 0, "dwell": []}
        )
        row["walk_bys"] += 1
        if p["is_look"]:
            row["looks"] += 1
            row["dwell"].append(p["dwell_look_s"])
    summary = []
    for ad_id, row in by_ad.items():
        walk_bys = row["walk_bys"]
        summary.append({
            "ad_id": ad_id,
            "walk_bys": walk_bys,
            "looks": row["looks"],
            "look_rate": row["looks"] / walk_bys if walk_bys else 0.0,
            "avg_look_duration_s": (
                statistics.mean(row["dwell"]) if row["dwell"] else 0.0
            ),
            "ad_seconds": ad_seconds.get(ad_id, 0.0),
        })
    summary.sort(key=lambda r: (-r["look_rate"], -r["walk_bys"], r["ad_id"]))
    return summary


# -------------------------------------------------------------- outputs

def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def write_outputs(data_dir, segments, passes, summary):
    """Write ad_segments.csv, passes.csv and ad_summary.csv."""
    segment_rows = [{
        "segment_id": f"seg_{i:03d}",
        "ad_id": s["ad_id"],
        "started_at": f"{s['started_at']:.3f}",
        "ended_at": f"{s['ended_at']:.3f}",
        "duration_s": f"{s['duration_s']:.3f}",
        "fingerprint_id": s["fingerprint_id"],
        "confidence": f"{s['confidence']:.3f}",
        "source": "auto",
    } for i, s in enumerate(segments)]
    pass_rows = [{
        "pass_id": p["pass_id"],
        "entered_at": f"{p['entered_at']:.3f}",
        "exited_at": f"{p['exited_at']:.3f}",
        "ad_id": p["ad_id"],
        "is_look": str(p["is_look"]).lower(),
        "dwell_look_s": f"{p['dwell_look_s']:.3f}",
        "scored": str(p["scored"]).lower(),
    } for p in passes]
    summary_rows = [{
        "ad_id": r["ad_id"],
        "walk_bys": r["walk_bys"],
        "looks": r["looks"],
        "look_rate": f"{r['look_rate']:.4f}",
        "avg_look_duration_s": f"{r['avg_look_duration_s']:.3f}",
        "ad_seconds": f"{r['ad_seconds']:.3f}",
    } for r in summary]
    paths = {
        "ad_segments": os.path.join(data_dir, "ad_segments.csv"),
        "passes": os.path.join(data_dir, "passes.csv"),
        "ad_summary": os.path.join(data_dir, "ad_summary.csv"),
    }
    write_csv(paths["ad_segments"],
              ["segment_id", "ad_id", "started_at", "ended_at",
               "duration_s", "fingerprint_id", "confidence", "source"],
              segment_rows)
    write_csv(paths["passes"],
              ["pass_id", "entered_at", "exited_at", "ad_id",
               "is_look", "dwell_look_s", "scored"], pass_rows)
    write_csv(paths["ad_summary"],
              ["ad_id", "walk_bys", "looks", "look_rate",
               "avg_look_duration_s", "ad_seconds"], summary_rows)
    return paths


# ------------------------------------------------------- frame processing

def list_frames(frames_dir):
    """Sorted frame files, with the numeric index parsed from the name."""
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(
            f"Frames directory not found: {frames_dir!r}. Sample frames with: "
            'ffmpeg -i data/demo_clip.mp4 -vf "fps=2,scale=1280:720" '
            "frames/frame_%04d.jpg"
        )
    entries = []
    for name in sorted(os.listdir(frames_dir)):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            stem = os.path.splitext(name)[0]
            digits = "".join(ch for ch in stem if ch.isdigit())
            entries.append((int(digits) if digits else len(entries),
                            os.path.join(frames_dir, name)))
    if not entries:
        raise FileNotFoundError(f"No frame images in {frames_dir!r}.")
    return entries


def roi_fingerprint(frame):
    """Fingerprint for the billboard ROI of one frame (numpy/cv2)."""
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    x0, y0, x1, y1 = config.BILLBOARD_ROI
    roi = frame[int(y0 * height):int(y1 * height),
                int(x0 * width):int(x1 * width)]
    hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    total = float(hist.sum())
    hist = (hist / total).flatten().tolist() if total > 0 else [0.0] * 512
    mean = np.mean(roi.reshape(-1, 3), axis=0).tolist() if roi.size else [0.0, 0.0, 0.0]
    return (hist, mean)


def _polygon_bbox(polygon, width, height):
    xs = [x * width for x, _ in polygon]
    ys = [y * height for _, y in polygon]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _in_zone(bbox, width, height):
    """Centre in gaze polygon and not in the red-steps exclusion."""
    cx = ((bbox[0] + bbox[2]) / 2.0) / width
    cy = ((bbox[1] + bbox[3]) / 2.0) / height
    if not point_in_polygon(cx, cy, config.GAZE_POLYGON):
        return False
    rx0, ry0, rx1, ry1 = config.RED_STEPS_RECT
    if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
        return False
    return True


def _best_pose(pose_list, bbox):
    best, best_iou = None, 0.0
    for pose in pose_list:
        iou = box_iou(bbox, pose["bbox"])
        if iou > best_iou:
            best, best_iou = pose, iou
    return best if best_iou > 0.3 else None


def build_yolo_estimators():
    """Build the real YOLO11n detector and YOLO11n-pose estimators."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for the real detector. "
            "Install requirements.txt in the project venv."
        ) from exc
    detect_model = YOLO("yolo11n.pt")
    pose_model = YOLO("yolo11n-pose.pt")

    def detector(frame_crop):
        results = detect_model.track(
            frame_crop, persist=True, classes=[0], verbose=False)
        detections = []
        for box in results[0].boxes:
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            track_id = int(box.id[0]) if box.id is not None else None
            detections.append({
                "track_id": track_id,
                "bbox": tuple(xyxy),
                "conf": float(box.conf[0]) if box.conf is not None else 0.0,
            })
        return detections

    def pose_estimator(frame):
        results = pose_model(frame, verbose=False)
        poses = []
        result = results[0]
        if result.keypoints is None:
            return poses
        keypoints_xy = result.keypoints.xy.tolist()
        keypoints_conf = result.keypoints.conf.tolist()
        boxes = result.boxes.xyxy.tolist() if result.boxes is not None else []
        for i, keypoints in enumerate(keypoints_xy):
            poses.append({
                "bbox": tuple(boxes[i]) if i < len(boxes) else (0, 0, 0, 0),
                "keypoints": [(float(x), float(y), float(keypoints_conf[i][j]))
                              for j, (x, y) in enumerate(keypoints)],
            })
        return poses

    return detector, pose_estimator


def _annotate_frame(frame, width, height, boxes, look_by_track):
    """Draw polygon, ROI and boxes; blur head regions. Returns frame."""
    import cv2
    import numpy as np

    polygon = np.array(
        [[int(x * width), int(y * height)] for x, y in config.GAZE_POLYGON],
        dtype=np.int32)
    cv2.polylines(frame, [polygon], True, (255, 255, 0), 2)
    x0, y0, x1, y1 = config.BILLBOARD_ROI
    cv2.rectangle(frame, (int(x0 * width), int(y0 * height)),
                  (int(x1 * width), int(y1 * height)), (0, 165, 255), 2)
    for track_id, bbox in boxes:
        bx0, by0, bx1, by1 = (int(v) for v in bbox)
        colour = (0, 200, 0) if look_by_track.get(track_id) else (180, 180, 180)
        head_h = int((by1 - by0) * 0.30)
        if head_h > 8 and bx1 > bx0 and by1 > by0:
            head = frame[by0:by0 + head_h, bx0:bx1]
            frame[by0:by0 + head_h, bx0:bx1] = cv2.GaussianBlur(
                head, (31, 31), 0)
        cv2.rectangle(frame, (bx0, by0), (bx1, by1), colour, 2)
    return frame


def run_pipeline(frames_dir, data_dir=None, fps=None, detector=None,
                 pose_estimator=None, upload_fn=None, clip_path=None,
                 limit=None, input_source="file",
                 debug_frames_dir=None, thumbnails_dir=None):
    """Run the full pipeline on a frames directory.

    Writes CSVs + run_meta.json locally first, then uploads the
    CSVs (and the clip, when given) to S3. The upload is
    required: if it raises, the error propagates and the run
    does not count as done. Pass a stub ``upload_fn`` in tests.
    Returns a dict of output paths, summary rows and S3 URIs.
    """
    import cv2

    from storage import upload_to_s3

    data_dir = config.DATA_DIR if data_dir is None else data_dir
    fps = config.SAMPLE_FPS if fps is None else fps
    debug_frames_dir = config.DEBUG_FRAMES_DIR if debug_frames_dir is None else debug_frames_dir
    thumbnails_dir = config.THUMBNAILS_DIR if thumbnails_dir is None else thumbnails_dir
    if detector is None or pose_estimator is None:
        real_detector, real_pose = build_yolo_estimators()
        detector = real_detector if detector is None else detector
        pose_estimator = real_pose if pose_estimator is None else pose_estimator
    if upload_fn is None:
        upload_fn = upload_to_s3

    entries = list_frames(frames_dir)
    if limit is not None:
        entries = entries[:limit]
    frame_numbers = [n for n, _ in entries]
    expected_frames = (max(frame_numbers) - min(frame_numbers) + 1) if frame_numbers else 0

    fingerprints, frame_times = [], []
    frame_records = []
    track_obs, track_times, track_frame_idx = {}, {}, {}
    prev_fallback, next_track_id = {}, 1
    processed = 0
    for position, (frame_no, path) in enumerate(entries):
        frame = cv2.imread(path)
        if frame is None:
            continue
        processed += 1
        height, width = frame.shape[:2]
        frame_time = frame_no / fps
        frame_times.append(frame_time)
        fingerprints.append(roi_fingerprint(frame))
        crop_x0, crop_y0, crop_x1, crop_y1 = _polygon_bbox(
            config.GAZE_POLYGON, width, height)
        crop = frame[crop_y0:crop_y1, crop_x0:crop_x1]
        detections = detector(crop)
        offset = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            bbox = (x1 + crop_x0, y1 + crop_y0, x2 + crop_x0, y2 + crop_y0)
            if _in_zone(bbox, width, height):
                offset.append({**det, "bbox": bbox})
        poses = pose_estimator(frame)
        boxes_needing_id = [d for d in offset if d.get("track_id") is None]
        if boxes_needing_id:
            assignments, next_track_id = assign_track_ids(
                prev_fallback, [d["bbox"] for d in boxes_needing_id],
                next_track_id)
            prev_fallback = {}
            for (box_idx, track_id), det in zip(assignments, boxes_needing_id):
                det["track_id"] = track_id
                prev_fallback[track_id] = det["bbox"]
        frame_boxes = []
        for det in offset:
            track_id = det["track_id"]
            bbox = det["bbox"]
            pose = _best_pose(poses, bbox)
            keypoints = pose["keypoints"] if pose else None
            g1, g2, scored = evaluate_gates(keypoints, bbox[3] - bbox[1])
            track_obs.setdefault(track_id, []).append((g1, g2, scored))
            track_times.setdefault(track_id, []).append(frame_time)
            track_frame_idx.setdefault(track_id, []).append(len(frame_times) - 1)
            frame_boxes.append((track_id, bbox))
        frame_records.append((path, frame_boxes))

    segments = assign_ad_ids(segment_fingerprints(
        fingerprints, frame_times, fps))
    frame_ad = [None] * len(frame_times)
    for segment in segments:
        for i in range(segment["start_idx"], segment["end_idx"]):
            frame_ad[i] = segment["ad_id"]

    passes = []
    look_by_track = {}
    for order, track_id in enumerate(sorted(track_obs, key=lambda t: min(track_times[t]))):
        observations = track_obs[track_id]
        is_look, dwell = evaluate_pass(observations, fps)
        look_by_track[track_id] = is_look
        passes.append({
            "pass_id": f"pass_{order + 1:03d}",
            "entered_at": min(track_times[track_id]),
            "exited_at": max(track_times[track_id]),
            "ad_id": majority_ad([frame_ad[i] for i in track_frame_idx[track_id]]),
            "is_look": is_look,
            "dwell_look_s": dwell,
            "scored": any(scored for _, _, scored in observations),
        })
    summary = aggregate_passes(passes, segments)
    paths = write_outputs(data_dir, segments, passes, summary)

    total_walk_bys = len(passes)
    total_looks = sum(1 for p in passes if p["is_look"])
    coverage = processed / expected_frames if expected_frames else 0.0
    meta = {
        "input_source": input_source,
        "frames_processed": processed,
        "frames_expected": expected_frames,
        "coverage": round(coverage, 4),
        "total_walk_bys": total_walk_bys,
        "total_looks": total_looks,
        "overall_look_rate": round(total_looks / total_walk_bys, 4) if total_walk_bys else 0.0,
        "total_ad_seconds": round(sum(s["duration_s"] for s in segments), 3),
        "s3_paths": {},
    }
    meta_path = os.path.join(data_dir, "run_meta.json")
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)

    os.makedirs(debug_frames_dir, exist_ok=True)
    os.makedirs(thumbnails_dir, exist_ok=True)
    seen_thumbnails = set()
    for segment in segments:
        middle = (segment["start_idx"] + segment["end_idx"] - 1) // 2
        path, boxes = frame_records[middle]
        frame = cv2.imread(path)
        height, width = frame.shape[:2]
        annotated = _annotate_frame(frame, width, height, boxes, look_by_track)
        cv2.imwrite(os.path.join(
            debug_frames_dir, os.path.basename(path)), annotated)
        if segment["ad_id"] not in seen_thumbnails:
            seen_thumbnails.add(segment["ad_id"])
            x0, y0, x1, y1 = config.BILLBOARD_ROI
            roi = frame[int(y0 * height):int(y1 * height),
                        int(x0 * width):int(x1 * width)]
            cv2.imwrite(os.path.join(
                thumbnails_dir, f"{segment['ad_id']}.jpg"), roi)

    uploads = [
        (paths["ad_summary"], f"{config.S3_RESULTS_PREFIX}/ad_summary.csv"),
        (paths["passes"], f"{config.S3_RESULTS_PREFIX}/passes.csv"),
        (paths["ad_segments"], f"{config.S3_RESULTS_PREFIX}/ad_segments.csv"),
    ]
    if clip_path and os.path.exists(clip_path):
        uploads.append((clip_path,
                        f"{config.S3_CLIPS_PREFIX}/{os.path.basename(clip_path)}"))
    s3_paths = {}
    for local_path, s3_key in uploads:
        s3_paths[os.path.basename(local_path)] = upload_fn(local_path, s3_key)
    meta["s3_paths"] = s3_paths
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)
    return {"paths": paths, "meta_path": meta_path, "summary": summary,
            "passes": passes, "segments": segments, "s3_paths": s3_paths,
            "meta": meta}


def main():
    """CLI: python pipeline.py --frames frames --limit 20."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyse sampled frames.")
    parser.add_argument("--frames", default=config.FRAMES_DIR)
    parser.add_argument("--data", default=config.DATA_DIR)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max frames (20 = 10s slice at 2fps).")
    parser.add_argument("--clip", default=None,
                        help="Source clip path, uploaded to S3 if present.")
    parser.add_argument("--source", default="file",
                        choices=["file", "stream"])
    args = parser.parse_args()
    result = run_pipeline(args.frames, data_dir=args.data,
                          limit=args.limit, clip_path=args.clip,
                          input_source=args.source)
    print(f"Wrote {result['paths']['ad_summary']}")
    print(f"Walk-bys: {result['meta']['total_walk_bys']} "
          f"Looks: {result['meta']['total_looks']} "
          f"Coverage: {result['meta']['coverage']}")
    for name, uri in result["s3_paths"].items():
        print(f"S3 {name}: {uri}")


if __name__ == "__main__":
    main()
