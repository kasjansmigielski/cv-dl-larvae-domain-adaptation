#!/usr/bin/env python3
"""Download raw recordings from S3 for the YOLO prediction pipeline.

S3 credentials and bucket/prefix are read from a local .env file
(NEVER hard-coded). Copy .env.example to .env and fill in your values.

Usage:
  python data/download_videos.py            # both sessions
  python data/download_videos.py s1         # session I only
  python data/download_videos.py s2         # session II only
"""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
ACCESS_KEY = os.getenv("S3_AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("S3_AWS_SECRET_ACCESS_KEY")
BUCKET = os.getenv("S3_BUCKET_NAME")
# Base S3 prefix that contains the per-session subfolders (e.g. "project/exp001/").
S3_BASE_PREFIX = os.getenv("S3_BASE_PREFIX", "")

# Files to download per session.
FILES_TO_DOWNLOAD = {
    "s1": {
        "prefix": f"s3://{BUCKET}/{S3_BASE_PREFIX}2025_05_14/",
        "files": [
            "control.mp4",
            "pbs.mp4",
            "coli_2x10_8.mp4",
            "coli_5x10_7.mp4",
            "coli_5x10_8.mp4",
        ],
    },
    "s2": {
        "prefix": f"s3://{BUCKET}/{S3_BASE_PREFIX}2025_12_04/",
        "files": [
            "control.mov",
            "pbs.mov",
            "coli_2x10_8.mov",
            "coli_5x10_7.mov",
            "coli_5x10_8.mov",
        ],
    },
}

OUTPUT_BASE = REPO_ROOT / "data" / "raw_videos"


def download_file(session_key: str, s3_prefix: str, filename: str):
    """Download a single file from S3 using the AWS CLI."""
    output_dir = OUTPUT_BASE / session_key
    output_dir.mkdir(parents=True, exist_ok=True)

    dest = output_dir / filename
    if dest.exists() and dest.stat().st_size > 1_000_000:
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  [skip] exists: {filename} ({size_mb:.0f} MB)")
        return True

    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = ACCESS_KEY
    env["AWS_SECRET_ACCESS_KEY"] = SECRET_KEY

    s3_url = s3_prefix + filename
    cmd = [
        "aws", "s3", "cp", s3_url, str(dest),
        "--endpoint-url", ENDPOINT_URL,
    ]

    print(f"  Downloading: {filename} ...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        return False
    else:
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  [ok] downloaded: {filename} ({size_mb:.0f} MB)")
        return True


def download_session(session_key: str):
    """Download the selected recordings from a session."""
    cfg = FILES_TO_DOWNLOAD[session_key]
    prefix = cfg["prefix"]
    files = cfg["files"]

    print(f"\n{'='*60}")
    print(f"Session {session_key.upper()}: downloading {len(files)} recordings")
    print(f"Source: {prefix}")
    print(f"Target folder: {OUTPUT_BASE / session_key}")
    print(f"{'='*60}")

    success = 0
    for f in files:
        if download_file(session_key, prefix, f):
            success += 1

    print(f"\n  Downloaded {success}/{len(files)} files for session {session_key}")
    return success == len(files)


if __name__ == "__main__":
    if not all([ENDPOINT_URL, ACCESS_KEY, SECRET_KEY, BUCKET]):
        print("ERROR: missing S3 configuration in .env")
        print("Required: S3_ENDPOINT_URL, S3_AWS_ACCESS_KEY_ID, "
              "S3_AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME")
        sys.exit(1)

    sessions = sys.argv[1:] if len(sys.argv) > 1 else FILES_TO_DOWNLOAD.keys()

    all_ok = True
    for key in sessions:
        if key not in FILES_TO_DOWNLOAD:
            print(f"Unknown session: {key}. Available: {list(FILES_TO_DOWNLOAD.keys())}")
            continue
        if not download_session(key):
            all_ok = False

    print(f"\n{'='*60}")
    if all_ok:
        print(f"Done. Recordings downloaded to: {OUTPUT_BASE}")
    else:
        print("WARNING: not all files were downloaded")
    print(f"{'='*60}")
