#!/usr/bin/env python3
"""Download raw recordings from S3 into the local records/ folder.

Session mapping:
  - Session I  (S1): S3 <prefix>/2025_05_14/  ->  records/session1/
  - Session II (S2): S3 <prefix>/2025_12_04/  ->  records/session2/

S3 credentials and bucket/prefix are read from a local .env file
(NEVER hard-coded). Copy .env.example to .env and fill in your values.

Usage:
  python data/download_records_from_s3.py            # both sessions
  python data/download_records_from_s3.py s1         # session I only
  python data/download_records_from_s3.py s2         # session II only
"""

import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
ACCESS_KEY = os.getenv("S3_AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("S3_AWS_SECRET_ACCESS_KEY")
BUCKET = os.getenv("S3_BUCKET_NAME")
# Base S3 prefix that contains the per-session subfolders (e.g. "project/exp001/").
S3_BASE_PREFIX = os.getenv("S3_BASE_PREFIX", "")

RECORDS_BASE = REPO_ROOT / "records"

# Base file names to skip per session (without extension).
SKIP = {"etanol_70proc"}

# Session configuration: S3 sub-prefix + local folder + recording extension.
SESSIONS = {
    "s1": {
        "s3_prefix": f"{S3_BASE_PREFIX}2025_05_14/",
        "local_dir": RECORDS_BASE / "session1",
        "ext": ".mp4",
    },
    "s2": {
        "s3_prefix": f"{S3_BASE_PREFIX}2025_12_04/",
        "local_dir": RECORDS_BASE / "session2",
        "ext": ".mov",
    },
}


def make_client():
    """Create an S3 client with extended timeouts (large files, ~1.6-2 GB)."""
    cfg = Config(
        connect_timeout=30,
        read_timeout=300,
        retries={"max_attempts": 3, "mode": "standard"},
        s3={"addressing_style": "path"},
    )
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=cfg,
    )


def list_session_videos(s3, prefix: str, ext: str):
    """Return a list of (key, size) for video files with the given extension under a prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    out = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(ext.lower()):
                out.append((key, obj["Size"]))
    return sorted(out)


def human_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.0f} MB"


def download_one(s3, key: str, size: int, dest: Path):
    """Download a single file with progress; skip if it already exists at full size."""
    if dest.exists() and dest.stat().st_size == size:
        print(f"  [skip] already downloaded: {dest.name} ({human_mb(size)})")
        return True

    print(f"  Downloading: {dest.name} ({human_mb(size)}) ...")
    downloaded = {"n": 0, "last": 0}

    def progress(chunk):
        # Log progress roughly every 10% of the file.
        downloaded["n"] += chunk
        pct = downloaded["n"] / size * 100 if size else 0
        if pct - downloaded["last"] >= 10:
            downloaded["last"] = pct
            print(f"    {pct:5.1f}%  ({human_mb(downloaded['n'])}/{human_mb(size)})")

    try:
        s3.download_file(BUCKET, key, str(dest), Callback=progress)
    except Exception as e:  # noqa: BLE001 - report any download error
        print(f"  ERROR downloading {dest.name}: {e}")
        # Remove the partial file so a re-run retries cleanly.
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False

    ok = dest.exists() and dest.stat().st_size == size
    print(f"  [ok] downloaded: {dest.name} ({human_mb(dest.stat().st_size)})" if ok
          else f"  ERROR: incomplete file {dest.name}")
    return ok


def download_session(s3, session_key: str) -> bool:
    cfg = SESSIONS[session_key]
    prefix, local_dir, ext = cfg["s3_prefix"], cfg["local_dir"], cfg["ext"]
    local_dir.mkdir(parents=True, exist_ok=True)

    videos = list_session_videos(s3, prefix, ext)
    # Filter out files listed in SKIP.
    to_get = [(k, sz) for (k, sz) in videos if Path(k).stem not in SKIP]
    skipped = [k for (k, _) in videos if Path(k).stem in SKIP]

    print(f"\n{'=' * 64}")
    print(f"Session {session_key.upper()}  ({prefix})")
    print(f"  -> target folder: {local_dir.relative_to(REPO_ROOT)}")
    print(f"  Found {len(videos)} recordings, {len(to_get)} to download.")
    if skipped:
        print(f"  Skipped (excluded from study): {[Path(k).name for k in skipped]}")
    print(f"{'=' * 64}")

    success = 0
    for key, size in to_get:
        dest = local_dir / Path(key).name
        if download_one(s3, key, size, dest):
            success += 1

    print(f"\n  Downloaded {success}/{len(to_get)} recordings for session {session_key.upper()}")
    return success == len(to_get)


def main():
    if not all([ENDPOINT_URL, ACCESS_KEY, SECRET_KEY, BUCKET]):
        print("ERROR: missing S3 configuration in .env")
        print("Required: S3_ENDPOINT_URL, S3_AWS_ACCESS_KEY_ID, "
              "S3_AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME")
        sys.exit(1)

    requested = [a.lower() for a in sys.argv[1:]] or list(SESSIONS.keys())
    unknown = [r for r in requested if r not in SESSIONS]
    if unknown:
        print(f"Unknown sessions: {unknown}. Available: {list(SESSIONS.keys())}")
        sys.exit(1)

    s3 = make_client()
    all_ok = True
    for key in requested:
        if not download_session(s3, key):
            all_ok = False

    print(f"\n{'=' * 64}")
    if all_ok:
        print(f"DONE. Recordings in: {RECORDS_BASE.relative_to(REPO_ROOT)}/")
        print("Next step: notebooks/cv_pipeline/02_dish_isolation.ipynb")
    else:
        print("WARNING: not all files downloaded correctly - re-run to finish.")
    print(f"{'=' * 64}")
    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
