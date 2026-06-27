# Data

Raw video recordings are **not** included in this repository (they are large and
are git-ignored). Use the scripts here to download them from S3-compatible storage.

## Setup

Set the S3 credentials first (see the project root `.env.example`):

```bash
cp ../.env.example ../.env
# edit ../.env with your endpoint, keys, bucket and prefix
```

## Scripts

- `download_records_from_s3.py` — downloads the full set of raw session recordings
  using `boto3`. Sessions are stored under `records/session1/` and `records/session2/`.
- `download_videos.py` — downloads a specific subset of recordings used by the YOLO
  pipeline (uses the AWS CLI under the hood).

## Expected layout after download

```
records/
├── session1/   # session 1 recordings (.mp4)
└── session2/   # session 2 recordings (.mov)
```