"""Streamlit dashboard: one page, no auth, no multi-user.

Reads the CSVs written by pipeline.py. If no results exist yet,
it shows the input controls and instructions instead of fake
numbers. The leaderboard is the hero; every look-rate is shown
with its walk-by count.
"""

import os

import config
import ingest


def _read_csv(path):
    import pandas as pd

    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def _load_results(data_dir):
    summary = _read_csv(os.path.join(data_dir, "ad_summary.csv"))
    segments = _read_csv(os.path.join(data_dir, "ad_segments.csv"))
    return summary, segments


def main():
    import streamlit as st

    st.set_page_config(page_title="Billboard Attention Tracker", layout="wide")
    st.title("Billboard Attention Tracker")
    st.caption(
        "The product takes a stream link, captures a window, and analyzes "
        "it. For this demo we pre-captured a 3-minute window from that "
        "same stream for reliability — the analysis pipeline is identical."
    )

    data_dir = config.DATA_DIR
    summary, _segments = _load_results(data_dir)
    meta = {}
    meta_path = os.path.join(data_dir, "run_meta.json")
    if os.path.exists(meta_path):
        import json

        with open(meta_path) as handle:
            meta = json.load(handle)

    with st.sidebar:
        st.header("Input")
        source = st.radio("Source", ["Demo clip", "Stream URL"],
                          index=0 if meta.get("input_source", "file") == "file"
                          else 1)
        stream_url = st.text_input("Stream URL", config.INPUT["stream_url"])
        if st.button("Capture 3 min"):
            try:
                clip = ingest.get_clip(
                    {"source": "stream", "stream_url": stream_url,
                     "capture_seconds": config.INPUT["capture_seconds"]},
                    fallback_path=config.INPUT["file_path"])
                st.success(f"Captured: {clip}")
            except Exception as exc:  # noqa: BLE001 - show in UI
                st.error(f"Capture failed: {exc}")
        if st.button("Use 3-min demo clip"):
            st.info(f"Demo clip: {config.INPUT['file_path']}. "
                    "Run pipeline.py on its sampled frames to refresh results.")

    total_walk_bys = int(summary["walk_bys"].sum()) if not summary.empty else 0
    total_looks = int(summary["looks"].sum()) if not summary.empty else 0
    overall = total_looks / total_walk_bys if total_walk_bys else 0.0
    badge = "Stream URL" if meta.get("input_source") == "stream" else "Demo Clip"
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Input", badge)
    col2.metric("Walk-bys", total_walk_bys)
    col3.metric("Looks", total_looks)
    col4.metric("Overall look-rate", f"{overall:.1%}")
    col5.metric("Coverage", f"{meta.get('coverage', 0):.0%}")
    if meta.get("s3_paths"):
        st.caption("S3: " + ", ".join(meta["s3_paths"].values()))

    st.header("Per-ad leaderboard")
    if summary.empty:
        st.info("No analysis results yet. Run: "
                "python pipeline.py --frames frames")
    else:
        display = summary.copy()
        display["thumbnail"] = display["ad_id"].apply(
            lambda ad: os.path.join(config.THUMBNAILS_DIR, f"{ad}.jpg"))
        st.dataframe(display, width="stretch")
        if os.path.isdir(config.THUMBNAILS_DIR):
            thumbs = [os.path.join(config.THUMBNAILS_DIR, f)
                      for f in sorted(os.listdir(config.THUMBNAILS_DIR))
                      if f.endswith(".jpg")]
            if thumbs:
                st.image(thumbs, caption=[os.path.basename(t) for t in thumbs],
                         width=200)
        st.bar_chart(summary.set_index("ad_id")["look_rate"])
        st.download_button(
            "Export per-ad summary (CSV)",
            summary.to_csv(index=False), "ad_summary.csv", "text/csv")

    st.header("Look-rate over time")
    if summary.empty:
        st.info("No time-series data yet.")
    else:
        st.line_chart(summary.set_index("ad_id")[["look_rate", "walk_bys"]])

    st.header("Proof frames")
    debug_dir = config.DEBUG_FRAMES_DIR
    frames = sorted(f for f in os.listdir(debug_dir)
                    if f.endswith(".jpg")) if os.path.isdir(debug_dir) else []
    if not frames:
        st.info("No annotated frames yet. They are written by pipeline.py "
                "with faces blurred (green = look, gray = walk-by).")
    else:
        index = st.slider("Frame", 0, len(frames) - 1, 0)
        st.image(os.path.join(debug_dir, frames[index]),
                 caption=frames[index])

    st.caption(
        "Look = facing toward billboard + head up + holds >=1s. "
        "Approximate attention proxy, not eye-tracking. "
        "Aggregate only: no face identification, no per-person data."
    )


if __name__ == "__main__":
    main()
