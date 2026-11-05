"""The only AWS call in the project: S3 upload / download via boto3.

REQUIRED, not optional. Local files are written first, but the
pipeline does not count as done until the S3 upload succeeds.
If AWS_BUCKET or AWS credentials are missing, fail fast with a
clear error instead of silently skipping AWS.
"""

import os


def require_s3_bucket() -> str:
    """Return AWS_BUCKET or raise a clear error."""
    bucket = os.environ.get("AWS_BUCKET")
    if not bucket:
        raise RuntimeError(
            "AWS_BUCKET env var is required. Set it before running pipeline.py."
        )
    return bucket


def upload_to_s3(local_path, s3_key) -> str:
    """Upload one local file to S3. Returns the s3:// URI.

    Raises if credentials or the bucket are wrong. ``boto3`` is
    imported lazily so modules that only need the bucket name do
    not require it at import time.
    """
    import boto3

    bucket = require_s3_bucket()
    boto3.client("s3").upload_file(local_path, bucket, s3_key)
    return f"s3://{bucket}/{s3_key}"


def parse_s3_uri(s3_uri):
    """Split 's3://bucket/key' into (bucket, key)."""
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Not an s3:// URI: {s3_uri!r}")
    rest = s3_uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"Not an s3:// URI with bucket and key: {s3_uri!r}")
    return bucket, key


def download_from_s3(s3_uri, local_path) -> str:
    """Download an S3 object to a local path. Returns local_path."""
    import boto3

    bucket, key = parse_s3_uri(s3_uri)
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    boto3.client("s3").download_file(bucket, key, local_path)
    return local_path
