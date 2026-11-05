"""Feed ingestion: stream (product) or file (demo).

The analysis pipeline is identical for both. This module only
produces a clip file. Stream mode captures a window with ffmpeg;
demo mode returns the local clip directly and skips the network.
"""

import os
import subprocess

import config
from storage import download_from_s3


def resolve_stream(stream_url):
    """Resolve a stream URL to a playable URL.

    YouTube: ``yt-dlp -g`` run once, do not stream through Python.
    Direct HLS (.m3u8) and RTSP URLs pass through unchanged.
    """
    if stream_url.endswith(".m3u8") or stream_url.startswith("rtsp://"):
        return stream_url
    result = subprocess.run(
        ["yt-dlp", "-g", stream_url],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"yt-dlp returned no playable URL for {stream_url!r}")
    return lines[0]


def capture_stream(hls_url, capture_seconds, out_path):
    """Capture a window from a stream URL to a clip file with ffmpeg."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-reconnect", "1", "-i", hls_url,
            "-t", str(capture_seconds), out_path,
        ],
        check=True,
    )
    return out_path


def get_clip(cfg, fallback_path=None, out_path="/tmp/capture.mp4"):
    """Return the path of a clip to analyse.

    ``cfg`` uses the keys in ``config.INPUT``: ``source`` is
    ``"stream"`` or ``"file"``. If stream capture fails and a
    fallback clip exists, return the fallback and log the gap;
    otherwise raise. Stream drops are coverage gaps, never
    silent interpolation: a failed capture is an error unless
    the caller explicitly supplies the demo fallback.
    """
    source = cfg.get("source", "file")
    if source == "file":
        file_path = cfg.get("file_path", config.INPUT["file_path"])
        if file_path.startswith("s3://"):
            return download_from_s3(file_path, "/tmp/demo_clip.mp4")
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Demo clip not found: {file_path!r}. Place the 3-minute "
                "recording at data/demo_clip.mp4 or set INPUT['file_path']."
            )
        return file_path
    if source == "stream":
        try:
            hls_url = resolve_stream(cfg["stream_url"])
            return capture_stream(
                hls_url, cfg.get("capture_seconds", 180), out_path
            )
        except Exception as exc:
            if fallback_path and os.path.exists(fallback_path):
                print(f"Stream capture failed ({exc}); falling back to demo clip.")
                return fallback_path
            raise RuntimeError(f"Stream capture failed: {exc}") from exc
    raise ValueError(f"Unknown INPUT source: {source!r}")


def main():
    """CLI: python ingest.py --source stream --url ... --seconds 180."""
    import argparse

    parser = argparse.ArgumentParser(description="Capture or locate a clip.")
    parser.add_argument("--source", choices=["stream", "file"],
                        default=config.INPUT["source"])
    parser.add_argument("--url", default=config.INPUT["stream_url"])
    parser.add_argument("--path", default=config.INPUT["file_path"])
    parser.add_argument("--seconds", type=int,
                        default=config.INPUT["capture_seconds"])
    parser.add_argument("--out", default="/tmp/capture.mp4")
    args = parser.parse_args()
    cfg = {
        "source": args.source,
        "stream_url": args.url,
        "file_path": args.path,
        "capture_seconds": args.seconds,
    }
    clip = get_clip(cfg, fallback_path=args.path, out_path=args.out)
    print(clip)


if __name__ == "__main__":
    main()
