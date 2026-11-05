# Billboard Attention Tracker — Hackathon MVP

Measure, from a single camera feed alone, how many people walk past one
digital billboard vs. how many appear to look at it — broken down by
which advertisement was showing at the time.

> The product takes a stream link, captures a window, and analyzes it.
> For this demo we pre-captured a 3-minute window from that same stream
> for reliability — the analysis pipeline is identical.

One stream, one billboard (Welcome to Times Square screen), approximate
attention via three coarse gates: facing toward the billboard, head up,
and holds. Noise is roughly constant across ads, so the *ranking* of ads
by look-rate is the useful signal, not the absolute look-rate.

## Setup (hackathon, ~20 min)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# ffmpeg + yt-dlp + AWS CLI: brew install ffmpeg yt-dlp awscli (macOS)
# or apt install ffmpeg + pip install yt-dlp
```

Python 3.11 is the spec target; this was developed on Python 3.12.

AWS is required, not optional. The pipeline fails fast without it —
it does not silently skip the upload:

```bash
export AWS_BUCKET=<your-bucket>
export AWS_REGION=us-east-1
aws configure  # or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
aws s3 ls s3://$AWS_BUCKET/  # must succeed before running pipeline
```

## Run

```bash
# 1. Get clip (product path) OR use demo clip
python ingest.py --source stream --url "https://www.youtube.com/watch?v=AdUw5RdyZxI" --seconds 180
# OR: python ingest.py --source file --path data/demo_clip.mp4

# Place the 3-minute recording at data/demo_clip.mp4 (180s, ~30fps, 1080p).

# 2. Sample frames
ffmpeg -i data/demo_clip.mp4 -vf "fps=2,scale=1280:720" frames/frame_%04d.jpg

# 3. Run pipeline on a 10s slice first, then the full clip
python pipeline.py --frames frames --limit 20
python pipeline.py --frames frames   # full 360 frames; ends with REQUIRED S3 upload

# 4. Dashboard
streamlit run app.py
```

Test order: the 10s slice (20 frames) must run end-to-end in under
2 minutes before running the full 3 minutes. Verify the S3 upload on
the 10s run too, so AWS is not a surprise at demo time.

## Layout

```text
config.py         # BILLBOARD_ROI, GAZE_POLYGON, thresholds, INPUT
ingest.py         # get_clip(): stream -> capture, or file / s3://
storage.py        # REQUIRED: upload_to_s3() via boto3
pipeline.py       # frames -> passes.csv, ad_segments.csv, ad_summary.csv -> S3
app.py            # Streamlit dashboard
data/demo_clip.mp4
frames/           # 360 JPGs at 2fps for the demo clip (generated, ignored)
data/*.csv        # outputs (generated, ignored)
requirements.txt
tests/            # unit tests: python -m unittest discover -s tests -t .
```

## Outputs

- `ad_segments.csv`: segment_id, ad_id, started_at, ended_at, duration_s,
  fingerprint_id, confidence, source. Ad identity is a visual cluster
  only — no brand or text is inferred.
- `passes.csv`: pass_id, entered_at, exited_at, ad_id (majority ad for
  the pass), is_look, dwell_look_s, scored.
- `ad_summary.csv`: ad_id, walk_bys, looks, look_rate,
  avg_look_duration_s, ad_seconds. Always read look-rate with walk-bys.
- `run_meta.json`: input source, coverage, totals, S3 paths.
- `debug_frames/`: one annotated frame per segment, faces blurred.

## Privacy

Aggregate only. No face recognition, no re-identification across visits,
no demographics, no tracking a person across days. Debug frames are
face-blurred. Raw clips are retained at most 7 days for debugging, then
deleted. Only aggregate events and blurred frames are retained.

Look-rate is "facing toward billboard + head up + holds" — an attention
proxy, not proof that eyes were on that exact screen.

## Cut order if behind

1. Auto ad-cut -> hardcode timestamps. 2. Pose -> bbox-only facing.
3. Tracking -> per-frame counts with simple dedup. 4. Never cut:
leaderboard + proof frame + required S3 upload.
