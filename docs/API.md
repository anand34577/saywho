# API reference

Base URL: `http://<host>:7860`. Interactive docs (try every call from the browser): **`/docs`** (Swagger UI),
`/redoc`, and the machine-readable schema at `/openapi.json` (use it to generate clients in any language).

| Method | Path | What | Auth |
|---|---|---|---|
| `GET` | `/api/v1/health` | Liveness: the process is up | public |
| `GET` | `/api/v1/ready` | Readiness: `200` once the model is loaded, `503` before | public |
| `GET` | `/api/v1/info` | Version, backend/accelerator, fallbacks, limits, defaults | public |
| `POST` | `/api/v1/diarize` | Upload a file and wait for the result | key |
| `POST` | `/api/v1/jobs` | Upload a file, get a job id immediately (`202`) | key |
| `GET` | `/api/v1/jobs` | List jobs (no results) | key |
| `GET` | `/api/v1/jobs/{id}` | Job status; includes the JSON result when `done` | key |
| `GET` | `/api/v1/jobs/{id}/result?format=` | Result as `json`, `rttm`, `csv` or `srt` | key |
| `DELETE` | `/api/v1/jobs/{id}` | Delete a finished job, or cancel a queued one | key |

**Sync or jobs?** `/diarize` holds the HTTP request open until the result is ready. That suits files up to roughly an
hour on a GPU/NPU (inference runs at 50-120x real time). For long files on a CPU, or behind proxies with short
timeouts, use `/jobs` and poll. Both run the same pipeline, and requests are processed one at a time on the
accelerator.

## Authentication

Off by default. With `API_KEY=<key>` in `.env`, every call marked *key* needs one of:

```
X-API-Key: <key>
Authorization: Bearer <key>
```

Missing or wrong key: `401`. Health, ready and info stay public for load balancers and monitoring.

## `POST /api/v1/diarize`

`multipart/form-data` fields:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `file` | file | required | Any audio or video file ffmpeg can read (wav, mp3, m4a, flac, ogg, opus, mp4, mkv, mov, webm, ...) |
| `threshold` | float 0.01-0.99 | `0.5` | Speech probability above which a speaker counts as active. Lower finds more (and quieter) speech |
| `min_segment` | float s | `0.1` | Drop segments shorter than this |
| `merge_gap` | float s | `0.3` | Merge two segments of the same speaker separated by less than this |
| `include_probabilities` | bool | `false` | Also return the raw per-frame probabilities (`activity`, large) |
| `format` | `json`/`rttm`/`csv`/`srt` | `json` | `json` returns the full result below; the others return that file as a download |

Defaults can be changed server-wide with `DEFAULT_THRESHOLD`, `DEFAULT_MIN_SEGMENT`, `DEFAULT_MERGE_GAP`.

### Response (`format=json`)

```json
{
  "filename": "meeting.mp4",
  "model": "nvidia/Nemotron-3-Diarization",
  "backend": "cuda",
  "accelerator": "NVIDIA GeForce GTX 1650 (CUDA)",
  "duration": 51.904,
  "inference_time": 0.474,
  "real_time_factor": 109.6,
  "processing_time": 0.583,
  "parameters": {"threshold": 0.5, "min_segment": 0.1, "merge_gap": 0.3},
  "num_speakers": 3,
  "speakers": [
    {"id": 0, "label": "Speaker 1", "channel": 0, "talk_time": 13.7, "share_of_speech": 0.384,
     "share_of_duration": 0.264, "segments": 7, "average_segment": 1.957, "longest_segment": 3.57,
     "first_heard": 0.07, "last_heard": 50.48, "overlap_time": 0.0, "interruptions": 0}
  ],
  "segments": [
    {"index": 0, "speaker": "Speaker 1", "speaker_id": 0, "start": 0.07, "end": 1.32, "duration": 1.25,
     "overlaps_with": []}
  ],
  "summary": {
    "duration": 51.904, "speech_time": 35.7, "speech_ratio": 0.688, "silence_time": 16.204,
    "overlap_time": 0.0, "overlap_ratio": 0.0, "speaker_turns": 7, "interruptions": 0,
    "average_gap": 1.378, "max_simultaneous_speakers": 1, "turns_per_minute": 8.092
  },
  "transitions": {"labels": ["Speaker 1", "Speaker 2", "Speaker 3"], "matrix": [[0, 2, 0], [0, 0, 2], [2, 0, 0]]}
}
```

(lists shortened). All times are in seconds.

- `speakers` are numbered in the order they are first heard. Labels are per file: "Speaker 1" in two files is not
  necessarily the same person.
- `segments` are sorted by start time. Overlapping speech gives overlapping segments with different speakers, and
  `overlaps_with` names them.
- `share_of_speech` is a fraction of the time anyone speaks; `overlap_ratio` is overlapped speech / speech.
- `transitions.matrix[a][b]` counts how often speaker `b` took the turn after speaker `a`.
- `timings`: where the processing time went, per stage (see below).
- With `include_probabilities=true`: `channels` (model output channels that are ever active) and
  `activity: {frame_sec, probs[frame][channel]}` (10 ms frames, max-pooled for files over ~25 min).

The web UI computes its segments with exactly the same rules, so for the same settings the API and the UI agree.

### Timing breakdown (`timings`)

Each result includes how long each stage took. The same breakdown is logged once per file, and the UI shows it under
the summary cards.

```json
"timings": {
  "decode":       {"wall": 3.24,  "cpu": 1.64,   "cores": 0.5},
  "features":     {"wall": 2.12,  "cpu": 7.31,   "cores": 3.4},
  "queue_wait":   0.0,
  "inference":    {"wall": 77.05, "cpu": 415.69, "cores": 5.4},
  "accelerator":  {"wall": 75.17, "calls": 69},
  "cpu_side":     {"wall": 1.88},
  "segmentation": {"wall": 0.04,  "cpu": 0.30,   "cores": 7.4},
  "total": 82.46
}
```

(a 31-minute MP3 on the `onnx` backend)

| Stage | What runs | Where |
|---|---|---|
| `decode` | ffmpeg: file → 16 kHz mono | CPU (one core per file) |
| `features` | log-mel spectrogram of the whole file | CPU (multi-threaded) |
| `queue_wait` | waiting for another request to release the model | — |
| `inference` | the model, chunk by chunk | backend |
| `accelerator` | time blocked inside the NPU/ONNX call, incl. transfers (`rknn`/`onnx` backends only) | NPU |
| `cpu_side` | `inference` minus `accelerator`: embedding, speaker cache, padding around each NPU call | CPU |
| `segmentation` | probabilities → segments and statistics | CPU |

`wall` is elapsed seconds. `cpu` is CPU seconds used by the server process, plus ffmpeg on Linux/macOS (on Windows,
ffmpeg's CPU time isn't counted). `cores` = `cpu / wall`, the average number of busy cores. These numbers are
approximate when several requests run at once.

### Export formats

- `rttm`: `SPEAKER <file> 1 <start> <duration> <NA> <NA> Speaker_1 <NA> <NA>` (standard diarization scoring format)
- `csv`: `speaker,start,end,duration`
- `srt`: subtitle blocks `[Speaker 1]`, which load into any video player or editor

## Jobs

```
POST /api/v1/jobs            -> 202 {"id": "...", "status": "queued", "queue_position": 0, "status_url": ..., "result_url": ...}
GET  /api/v1/jobs/{id}       -> {"status": "queued" | "running" | "done" | "error", ..., "result": {...} when done}
GET  /api/v1/jobs/{id}/result?format=srt
DELETE /api/v1/jobs/{id}     -> 204
```

`POST /jobs` takes the same fields as `/diarize` except `format`, which you choose when fetching the result. Jobs live
in memory: they are lost on restart and deleted `JOB_TTL_MINUTES` (default 60) after finishing. Fetching a result
before it is ready returns `409` with a `Retry-After` header. More than `MAX_QUEUED_JOBS` pending jobs returns `429`.

## Errors

Errors are JSON `{"detail": "..."}` (`detail` is a list of field errors for `422`).

| Code | When |
|---|---|
| `400` | Not decodable as audio, empty file, no audio stream, audio shorter than 0.5 s |
| `401` | API key missing or wrong |
| `404` | Unknown or expired job |
| `409` | Job result requested before it is done; deleting a running job |
| `413` | Upload over `MAX_UPLOAD_MB`, or audio longer than `MAX_AUDIO_MINUTES` |
| `422` | Invalid parameter (for example `threshold=2`, `format=xml`); for job results: the job failed |
| `429` | Too many pending jobs |
| `503` | Model still loading (`Retry-After` header) or failed to load (see `/api/v1/info` → `error`) |

## Examples

### curl

```bash
# JSON result
curl -F file=@meeting.mp4 http://localhost:7860/api/v1/diarize

# RTTM file with custom settings and an API key
curl -H "X-API-Key: $KEY" -F file=@meeting.mp4 -F threshold=0.4 -F format=rttm \
     -o meeting.rttm http://localhost:7860/api/v1/diarize

# job: submit, poll, fetch SRT
ID=$(curl -s -F file=@long.mp3 http://localhost:7860/api/v1/jobs | jq -r .id)
until [ "$(curl -s http://localhost:7860/api/v1/jobs/$ID | jq -r .status)" = done ]; do sleep 5; done
curl -o long.srt "http://localhost:7860/api/v1/jobs/$ID/result?format=srt"
```

Windows PowerShell: use `curl.exe` (not the `curl` alias) with the same arguments.

### Python

`examples/client.py` is a complete client (sync and job modes, all formats). The core of it:

```python
import requests

with open("meeting.mp4", "rb") as f:
    r = requests.post("http://localhost:7860/api/v1/diarize",
                      files={"file": f}, data={"threshold": 0.5},
                      headers={"X-API-Key": "KEY"})        # omit if no API_KEY is set
r.raise_for_status()
result = r.json()
for seg in result["segments"]:
    print(f"{seg['start']:.2f}-{seg['end']:.2f} {seg['speaker']}")
```

### JavaScript (browser or Node 18+)

```js
const form = new FormData();
form.append("file", fileInput.files[0]);          // Node: new Blob([fs.readFileSync(p)]), "name.wav"
form.append("threshold", "0.5");
const res = await fetch("http://localhost:7860/api/v1/diarize", {
  method: "POST", body: form, headers: { "X-API-Key": "KEY" },
});
if (!res.ok) throw new Error((await res.json()).detail);
const { segments, speakers, summary } = await res.json();
```

Browser apps on another origin need `CORS_ORIGINS` set on the server (for example `CORS_ORIGINS=https://app.example.com`).

## Legacy routes

`GET /api/info` and `POST /api/diarize` (v1.x) still work. `/api/diarize` returns the v2 result with
`include_probabilities=true`, a superset of the old response.
