"""Speaker probabilities -> segments, per-speaker statistics, summary and export formats.

Same rules as the web UI (static/index.html), so API results match what the UI shows for the same settings:
threshold each channel, merge gaps shorter than `merge_gap`, drop segments shorter than `min_segment`, and number
speakers by the order in which they are first heard.
"""
import csv
import io
import json

import numpy as np

EXPORT_FORMATS = ("json", "rttm", "csv", "srt")


def _r(x: float) -> float:
    return round(float(x), 3)


def find_segments(probs: np.ndarray, frame_sec: float, duration: float, threshold: float = 0.5,
                  min_segment: float = 0.1, merge_gap: float = 0.3) -> list[dict]:
    """Per-channel thresholding. Returns segments sorted by start, each with `speaker_id` in order of first appearance."""
    segs = []
    for c in range(probs.shape[1]):
        edges = np.diff(np.concatenate([[0], (probs[:, c] >= threshold).astype(np.int8), [0]]))
        runs = zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
        merged: list[list[float]] = []
        for s, e in runs:
            start, end = s * frame_sec, min(e * frame_sec, duration)
            if merged and start - merged[-1][1] < merge_gap:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        segs += [{"start": s, "end": e, "channel": c} for s, e in merged if e - s >= min_segment]
    segs.sort(key=lambda s: (s["start"], s["end"]))
    order: list[int] = []
    for s in segs:
        if s["channel"] not in order:
            order.append(s["channel"])
        s["speaker_id"] = order.index(s["channel"])
    return segs


def analyze(probs: np.ndarray, frame_sec: float, duration: float, threshold: float = 0.5,
            min_segment: float = 0.1, merge_gap: float = 0.3) -> dict:
    """Segments + per-speaker stats + conversation summary + turn-transition matrix."""
    segs = find_segments(probs, frame_sec, duration, threshold, min_segment, merge_gap)
    n = 1 + max((s["speaker_id"] for s in segs), default=-1)
    label = [f"Speaker {i + 1}" for i in range(n)]
    spk = [{"total": 0.0, "count": 0, "longest": 0.0, "first": None, "last": 0.0, "overlap": 0.0, "interrupts": 0,
            "channel": None} for _ in range(n)]
    for s in segs:
        p, d = spk[s["speaker_id"]], s["end"] - s["start"]
        p["total"] += d
        p["count"] += 1
        p["longest"] = max(p["longest"], d)
        p["first"] = s["start"] if p["first"] is None else p["first"]
        p["last"] = max(p["last"], s["end"])
        p["channel"] = s["channel"]

    # sweep line: speech (>= 1 active), overlap (>= 2 active), per-speaker overlapped time
    events = sorted([(s["start"], 1, s["speaker_id"]) for s in segs] + [(s["end"], -1, s["speaker_id"]) for s in segs],
                    key=lambda e: (e[0], e[1]))
    active: set[int] = set()
    prev = speech = overlap = 0.0
    max_concurrent = 0
    for t, kind, sp in events:
        dt = t - prev
        if len(active) >= 1:
            speech += dt
        if len(active) >= 2:
            overlap += dt
            for a in active:
                spk[a]["overlap"] += dt
        if kind == 1:
            active.add(sp)
        else:
            active.discard(sp)
        max_concurrent = max(max_concurrent, len(active))
        prev = t

    # which other speakers overlap each segment (segments are sorted by start)
    overlaps = [set() for _ in segs]
    for i, s in enumerate(segs):
        j = i + 1
        while j < len(segs) and segs[j]["start"] < s["end"]:
            o = segs[j]
            if o["speaker_id"] != s["speaker_id"] and o["end"] > s["start"]:
                overlaps[i].add(o["speaker_id"])
                overlaps[j].add(s["speaker_id"])
            j += 1

    # turns: consecutive segments of the same speaker (in start order) collapse into one
    turns: list[list] = []
    for s in segs:
        if turns and turns[-1][0] == s["speaker_id"]:
            turns[-1][2] = max(turns[-1][2], s["end"])
        else:
            turns.append([s["speaker_id"], s["start"], s["end"]])
    transitions = [[0] * n for _ in range(n)]
    interruptions, gaps = 0, []
    for prev_turn, turn in zip(turns, turns[1:]):
        transitions[prev_turn[0]][turn[0]] += 1
        gap = turn[1] - prev_turn[2]
        if gap < 0:
            interruptions += 1
            spk[turn[0]]["interrupts"] += 1
        else:
            gaps.append(gap)

    return {
        "num_speakers": n,
        "speakers": [
            {
                "id": i,
                "label": label[i],
                "channel": p["channel"],
                "talk_time": _r(p["total"]),
                "share_of_speech": _r(p["total"] / speech) if speech else 0.0,
                "share_of_duration": _r(p["total"] / duration) if duration else 0.0,
                "segments": p["count"],
                "average_segment": _r(p["total"] / p["count"]),
                "longest_segment": _r(p["longest"]),
                "first_heard": _r(p["first"]),
                "last_heard": _r(p["last"]),
                "overlap_time": _r(p["overlap"]),
                "interruptions": p["interrupts"],
            }
            for i, p in enumerate(spk)
        ],
        "segments": [
            {
                "index": k,
                "speaker": label[s["speaker_id"]],
                "speaker_id": s["speaker_id"],
                "start": _r(s["start"]),
                "end": _r(s["end"]),
                "duration": _r(s["end"] - s["start"]),
                "overlaps_with": [label[o] for o in sorted(overlaps[k])],
            }
            for k, s in enumerate(segs)
        ],
        "summary": {
            "duration": _r(duration),
            "speech_time": _r(speech),
            "speech_ratio": _r(speech / duration) if duration else 0.0,
            "silence_time": _r(max(0.0, duration - speech)),
            "overlap_time": _r(overlap),
            "overlap_ratio": _r(overlap / speech) if speech else 0.0,
            "speaker_turns": len(turns),
            "interruptions": interruptions,
            "average_gap": _r(sum(gaps) / len(gaps)) if gaps else 0.0,
            "max_simultaneous_speakers": max_concurrent,
            "turns_per_minute": _r(len(turns) / (duration / 60)) if duration else 0.0,
        },
        "transitions": {"labels": label, "matrix": transitions},
    }


# ---------------------------------------------------------------- exports


def _srt_time(x: float) -> str:
    ms = round(x * 1000)
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def export(result: dict, fmt: str) -> tuple[str, str]:
    """Render a diarization result as (text, media type)."""
    segs = result["segments"]
    base = (result.get("filename") or "audio").rsplit(".", 1)[0].replace(" ", "_") or "audio"
    if fmt == "json":
        return json.dumps(result, indent=2), "application/json"
    if fmt == "rttm":
        lines = [f"SPEAKER {base} 1 {s['start']:.3f} {s['duration']:.3f} <NA> <NA> {s['speaker'].replace(' ', '_')} <NA> <NA>"
                 for s in segs]
        return "\n".join(lines) + "\n", "text/plain"
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["speaker", "start", "end", "duration"])
        w.writerows([s["speaker"], f"{s['start']:.3f}", f"{s['end']:.3f}", f"{s['duration']:.3f}"] for s in segs)
        return buf.getvalue(), "text/csv"
    if fmt == "srt":
        blocks = [f"{k + 1}\n{_srt_time(s['start'])} --> {_srt_time(s['end'])}\n[{s['speaker']}]\n" for k, s in enumerate(segs)]
        return "\n".join(blocks), "application/x-subrip"
    raise ValueError(f"unknown format {fmt!r}; choose one of {', '.join(EXPORT_FORMATS)}")
