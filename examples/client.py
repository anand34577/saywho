"""Minimal Python client for the Nemotron Diarization API (needs `pip install requests`).

    python examples/client.py meeting.mp4                       # print speakers and segments
    python examples/client.py meeting.mp4 --format rttm -o out.rttm
    python examples/client.py long_recording.mp3 --job          # async job API, for long files
    python examples/client.py x.wav --url http://board:7860 --api-key KEY --threshold 0.4
"""
import argparse
import os
import sys
import time

import requests


def diarize(url, path, api_key=None, fmt="json", use_job=False, **params):
    headers = {"X-API-Key": api_key} if api_key else {}
    data = {k: str(v) for k, v in params.items() if v is not None}
    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f)}
        if not use_job:
            r = requests.post(f"{url}/api/v1/diarize", files=files, data={**data, "format": fmt}, headers=headers)
            r.raise_for_status()
            return r.json() if fmt == "json" else r.text
        r = requests.post(f"{url}/api/v1/jobs", files=files, data=data, headers=headers)
    r.raise_for_status()
    job = r.json()
    while job["status"] in ("queued", "running"):
        time.sleep(2)
        job = requests.get(url + job["status_url"], headers=headers).json()
    if job["status"] != "done":
        raise RuntimeError(f"job failed: {job['error']}")
    r = requests.get(url + job["result_url"], params={"format": fmt}, headers=headers)
    r.raise_for_status()
    requests.delete(url + job["status_url"], headers=headers)
    return r.json() if fmt == "json" else r.text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--url", default=os.environ.get("DIAR_URL", "http://127.0.0.1:7860"))
    ap.add_argument("--api-key", default=os.environ.get("DIAR_API_KEY"))
    ap.add_argument("--format", default="json", choices=["json", "rttm", "csv", "srt"])
    ap.add_argument("--job", action="store_true", help="use the async job API")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--min-segment", type=float)
    ap.add_argument("--merge-gap", type=float)
    ap.add_argument("-o", "--output", help="write the result here instead of printing a summary")
    a = ap.parse_args()

    result = diarize(a.url.rstrip("/"), a.file, a.api_key, a.format, a.job,
                     threshold=a.threshold, min_segment=a.min_segment, merge_gap=a.merge_gap)
    if a.output:
        import json

        with open(a.output, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, indent=2) if a.format == "json" else result)
        print(f"wrote {a.output}")
    elif a.format != "json":
        sys.stdout.write(result)
    else:
        print(f"{result['filename']}: {result['duration']:.1f} s, {result['num_speakers']} speakers "
              f"({result['accelerator']}, {result['real_time_factor']:.0f}x real time)")
        for s in result["speakers"]:
            print(f"  {s['label']}: {s['talk_time']:.1f} s ({s['share_of_speech'] * 100:.0f}% of speech), {s['segments']} segments")
        for s in result["segments"]:
            print(f"  {s['start']:8.2f} - {s['end']:8.2f}  {s['speaker']}")


if __name__ == "__main__":
    main()
