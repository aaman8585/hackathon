"""Unit tests for the pure pipeline logic, ingest and storage.

Uses only the standard library so these run without any vision
dependency. The synthetic end-to-end pipeline test needs OpenCV
and is skipped when cv2 is unavailable. Run with:

    python -m unittest discover -s tests -t .
"""

import csv
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import ingest
import pipeline
import storage


def _solid_fingerprint(hist_index, mean):
    hist = [0.0] * 512
    hist[hist_index] = 1.0
    return (hist, mean)


RED = _solid_fingerprint(0, (0, 0, 255))
BLUE = _solid_fingerprint(511, (255, 0, 0))


def _keypoints(nose, left_shoulder, right_shoulder):
    points = [(112.0, 150.0, 0.9)] * 17
    points[0] = nose
    points[5] = left_shoulder
    points[6] = right_shoulder
    return points


class TestConfigDefaults(unittest.TestCase):
    def test_roi_and_polygon_match_spec_v2(self):
        self.assertEqual(config.BILLBOARD_ROI, (0.24, 0.11, 0.44, 0.38))
        self.assertEqual(
            config.GAZE_POLYGON,
            [(0.05, 0.97), (0.65, 0.97), (0.55, 0.65), (0.10, 0.65)])
        self.assertEqual(config.HOLD_FRAMES, 3)
        self.assertEqual(config.MIN_PERSON_HEIGHT_PX, 40)
        self.assertEqual(config.INPUT["capture_seconds"], 180)


class TestPointInPolygon(unittest.TestCase):
    def test_inside_plaza(self):
        self.assertTrue(pipeline.point_in_polygon(
            0.35, 0.80, config.GAZE_POLYGON))

    def test_road_band_above_polygon_is_outside(self):
        self.assertFalse(pipeline.point_in_polygon(
            0.35, 0.60, config.GAZE_POLYGON))


class TestGates(unittest.TestCase):
    def test_facing_left_and_head_up_passes(self):
        keypoints = _keypoints((95.0, 130.0, 0.9), (102.0, 150.0, 0.9),
                               (122.0, 150.0, 0.9))
        self.assertEqual(pipeline.evaluate_gates(keypoints, 80),
                         (True, True, True))

    def test_facing_right_fails_g1_only(self):
        keypoints = _keypoints((129.0, 130.0, 0.9), (102.0, 150.0, 0.9),
                               (122.0, 150.0, 0.9))
        self.assertEqual(pipeline.evaluate_gates(keypoints, 80),
                         (False, True, True))

    def test_head_down_fails_g2_only(self):
        keypoints = _keypoints((95.0, 155.0, 0.9), (102.0, 150.0, 0.9),
                               (122.0, 150.0, 0.9))
        self.assertEqual(pipeline.evaluate_gates(keypoints, 80),
                         (True, False, True))

    def test_back_view_passes_g1_via_wide_shoulders(self):
        keypoints = _keypoints((112.0, 140.0, 0.1), (97.0, 150.0, 0.9),
                               (127.0, 150.0, 0.9))
        g1, g2, scored = pipeline.evaluate_gates(keypoints, 80)
        self.assertTrue(g1)
        self.assertFalse(g2)
        self.assertTrue(scored)

    def test_small_person_is_unscored(self):
        keypoints = _keypoints((95.0, 130.0, 0.9), (102.0, 150.0, 0.9),
                               (122.0, 150.0, 0.9))
        self.assertEqual(pipeline.evaluate_gates(keypoints, 30),
                         (False, False, False))

    def test_no_pose_is_not_a_look(self):
        self.assertEqual(pipeline.evaluate_gates(None, 80),
                         (False, False, True))


class TestEvaluatePass(unittest.TestCase):
    def test_three_consecutive_frames_is_a_look(self):
        is_look, dwell = pipeline.evaluate_pass(
            [(True, True, True)] * 3, fps=2)
        self.assertTrue(is_look)
        self.assertEqual(dwell, 1.5)

    def test_split_qualifying_frames_are_not_a_look(self):
        is_look, dwell = pipeline.evaluate_pass(
            [(True, True, True), (True, True, True), (False, False, True),
             (True, True, True)], fps=2)
        self.assertFalse(is_look)
        self.assertEqual(dwell, 1.5)

    def test_empty_pass_is_not_a_look(self):
        self.assertEqual(pipeline.evaluate_pass([], fps=2), (False, 0.0))


class TestSegmentation(unittest.TestCase):
    def test_recurring_creative_gets_same_ad_id(self):
        prints = [RED] * 4 + [BLUE] * 4 + [RED] * 4
        times = [i / 2 for i in range(len(prints))]
        segments = pipeline.assign_ad_ids(
            pipeline.segment_fingerprints(prints, times, fps=2))
        self.assertEqual([s["ad_id"] for s in segments],
                         ["ad_A", "ad_B", "ad_A"])
        self.assertEqual([s["duration_s"] for s in segments], [2.0, 2.0, 2.0])

    def test_subsecond_flicker_is_merged(self):
        prints = [RED] * 4 + [BLUE] + [RED] * 4
        times = [i / 2 for i in range(len(prints))]
        segments = pipeline.segment_fingerprints(prints, times, fps=2)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["duration_s"], 4.5)

    def test_leading_flicker_adopts_next_segment(self):
        prints = [BLUE] + [RED] * 4
        times = [i / 2 for i in range(len(prints))]
        segments = pipeline.assign_ad_ids(
            pipeline.segment_fingerprints(prints, times, fps=2))
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["ad_id"], "ad_A")
        self.assertEqual(segments[0]["started_at"], 0.0)

    def test_leading_flicker_keeps_next_segments_identity(self):
        prints = [BLUE] + [RED] * 4 + [BLUE] * 4
        times = [i / 2 for i in range(len(prints))]
        segments = pipeline.assign_ad_ids(
            pipeline.segment_fingerprints(prints, times, fps=2))
        self.assertEqual([s["ad_id"] for s in segments], ["ad_A", "ad_B"])

    def test_fingerprint_distance(self):
        self.assertEqual(pipeline.fingerprint_distance(RED, RED), 0.0)
        self.assertGreater(pipeline.fingerprint_distance(RED, BLUE), 0.30)


class TestTrackingFallback(unittest.TestCase):
    def test_same_box_keeps_track_id(self):
        assignments, next_id = pipeline.assign_track_ids(
            {1: (0, 0, 10, 10)}, [(0, 0, 10, 10)], next_track_id=2)
        self.assertEqual(assignments, [(0, 1)])
        self.assertEqual(next_id, 2)

    def test_far_box_gets_new_track_id(self):
        assignments, next_id = pipeline.assign_track_ids(
            {1: (0, 0, 10, 10)}, [(100, 100, 110, 110)], next_track_id=2)
        self.assertEqual(assignments, [(0, 2)])
        self.assertEqual(next_id, 3)


class TestAggregation(unittest.TestCase):
    def test_majority_ad_tie_goes_to_earliest(self):
        self.assertEqual(
            pipeline.majority_ad(["ad_A", "ad_B"]), "ad_A")

    def test_look_rate_and_dwell_only_count_looks(self):
        segments = [
            {"ad_id": "ad_A", "duration_s": 10.0},
            {"ad_id": "ad_B", "duration_s": 5.0},
        ]
        passes = [
            {"ad_id": "ad_A", "is_look": True, "dwell_look_s": 2.0},
            {"ad_id": "ad_A", "is_look": False, "dwell_look_s": 0.0},
            {"ad_id": "ad_A", "is_look": True, "dwell_look_s": 4.0},
        ]
        summary = pipeline.aggregate_passes(passes, segments)
        by_ad = {row["ad_id"]: row for row in summary}
        self.assertEqual(by_ad["ad_A"]["walk_bys"], 3)
        self.assertEqual(by_ad["ad_A"]["looks"], 2)
        self.assertAlmostEqual(by_ad["ad_A"]["look_rate"], 2 / 3)
        self.assertEqual(by_ad["ad_A"]["avg_look_duration_s"], 3.0)
        self.assertEqual(by_ad["ad_B"]["walk_bys"], 0)
        self.assertEqual(by_ad["ad_B"]["look_rate"], 0.0)
        self.assertEqual(by_ad["ad_B"]["ad_seconds"], 5.0)
        self.assertEqual(summary[0]["ad_id"], "ad_A")


class TestIngest(unittest.TestCase):
    def test_file_source_returns_existing_clip(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as clip:
            self.assertEqual(
                ingest.get_clip({"source": "file", "file_path": clip.name}),
                clip.name)

    def test_file_source_missing_clip_raises(self):
        with self.assertRaises(FileNotFoundError):
            ingest.get_clip({"source": "file",
                             "file_path": "/tmp/no-such-clip.mp4"})

    def test_unknown_source_raises(self):
        with self.assertRaises(ValueError):
            ingest.get_clip({"source": "rtmp"})

    def test_stream_failure_uses_fallback_clip(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as clip:
            with mock.patch.object(
                    ingest, "resolve_stream",
                    side_effect=RuntimeError("down")):
                self.assertEqual(
                    ingest.get_clip(
                        {"source": "stream", "stream_url": "https://x",
                         "capture_seconds": 1},
                        fallback_path=clip.name),
                    clip.name)

    def test_stream_failure_without_fallback_raises(self):
        with mock.patch.object(ingest, "resolve_stream",
                               side_effect=RuntimeError("down")):
            with self.assertRaises(RuntimeError):
                ingest.get_clip({"source": "stream",
                                 "stream_url": "https://x",
                                 "capture_seconds": 1})


class TestStorage(unittest.TestCase):
    def test_require_bucket_raises_without_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                storage.require_s3_bucket()

    def test_parse_s3_uri(self):
        self.assertEqual(storage.parse_s3_uri("s3://bucket/a/b.csv"),
                         ("bucket", "a/b.csv"))
        with self.assertRaises(ValueError):
            storage.parse_s3_uri("https://example.com/a")

    def test_upload_uses_boto3_client(self):
        calls = []

        class FakeClient:
            def upload_file(self, local_path, bucket, key):
                calls.append((local_path, bucket, key))

        fake_boto3 = mock.Mock()
        fake_boto3.client.return_value = FakeClient()
        with mock.patch.dict(os.environ, {"AWS_BUCKET": "my-bucket"}):
            with mock.patch.dict(sys.modules, {"boto3": fake_boto3}):
                uri = storage.upload_to_s3("/tmp/a.csv", "results/a.csv")
        self.assertEqual(uri, "s3://my-bucket/results/a.csv")
        self.assertEqual(calls, [("/tmp/a.csv", "my-bucket", "results/a.csv")])


@unittest.skipUnless(
    __import__("importlib").util.find_spec("cv2") is not None,
    "cv2 is required for the synthetic pipeline test")
class TestSyntheticPipeline(unittest.TestCase):
    def test_two_ads_one_look_end_to_end(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = os.path.join(tmp, "frames")
            os.makedirs(frames_dir)
            width, height = 320, 180
            x0, y0, x1, y1 = config.BILLBOARD_ROI
            roi = (int(x0 * width), int(y0 * height),
                   int(x1 * width), int(y1 * height))
            for i in range(1, 13):
                frame = np.full((height, width, 3), 128, dtype=np.uint8)
                colour = (0, 0, 255) if i <= 6 else (255, 0, 0)  # BGR
                frame[roi[1]:roi[3], roi[0]:roi[2]] = colour
                cv2.imwrite(os.path.join(frames_dir, f"frame_{i:04d}.jpg"),
                            frame)

            crop_x0 = int(min(x for x, _ in config.GAZE_POLYGON) * width)
            crop_y0 = int(min(y for _, y in config.GAZE_POLYGON) * height)
            person_crop = (92 - crop_x0, 123 - crop_y0,
                           132 - crop_x0, 173 - crop_y0)

            def detector(_crop):
                return [{"track_id": None, "bbox": person_crop, "conf": 0.9}]

            def pose_estimator(_frame):
                keypoints = [(112.0, 150.0, 0.9)] * 17
                keypoints[0] = (95.0, 130.0, 0.9)
                keypoints[5] = (102.0, 150.0, 0.9)
                keypoints[6] = (122.0, 150.0, 0.9)
                return [{"bbox": (92, 123, 132, 173),
                         "keypoints": keypoints}]

            uploads = []

            def fake_upload(local_path, s3_key):
                uploads.append((local_path, s3_key))
                return f"s3://bucket/{s3_key}"

            data_dir = os.path.join(tmp, "data")
            result = pipeline.run_pipeline(
                frames_dir, data_dir=data_dir, fps=2,
                detector=detector, pose_estimator=pose_estimator,
                upload_fn=fake_upload,
                debug_frames_dir=os.path.join(tmp, "debug"),
                thumbnails_dir=os.path.join(tmp, "thumbs"))
            self.assertEqual([s["ad_id"] for s in result["segments"]],
                             ["ad_A", "ad_B"])
            self.assertEqual(len(result["passes"]), 1)
            self.assertTrue(result["passes"][0]["is_look"])
            self.assertEqual(result["passes"][0]["ad_id"], "ad_A")
            self.assertEqual(result["meta"]["total_walk_bys"], 1)
            self.assertEqual(result["meta"]["total_looks"], 1)
            self.assertEqual(result["meta"]["coverage"], 1.0)
            self.assertEqual(len(uploads), 3)
            with open(os.path.join(data_dir, "ad_summary.csv")) as handle:
                rows = list(csv.DictReader(handle))
            by_ad = {row["ad_id"]: row for row in rows}
            self.assertEqual(by_ad["ad_A"]["walk_bys"], "1")
            self.assertEqual(by_ad["ad_A"]["looks"], "1")
            self.assertEqual(by_ad["ad_B"]["walk_bys"], "0")
            self.assertEqual(len(os.listdir(os.path.join(tmp, "debug"))), 2)
            self.assertEqual(len(os.listdir(os.path.join(tmp, "thumbs"))), 2)


if __name__ == "__main__":
    unittest.main()
