"""API and segmentation tests. They use a fake model, so they run in seconds without a GPU or model download.

    python -m unittest discover -s tests -v
"""
import io
import os
import sys
import time
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ENV_FILE"] = os.devnull  # ignore a local .env

from fastapi.testclient import TestClient  # noqa: E402

import app as server  # noqa: E402
import settings  # noqa: E402
from segmentation import analyze, export  # noqa: E402

SR = 16000


class FakeEngine:
    """Speaker A talks 1.0-3.0 s, speaker B 2.5-5.0 s (overlap 2.5-3.0), speaker A again 6.0-6.05 s (a blip)."""

    backend, accelerator, device, frame_sec = "fake", "Fake accelerator", "cpu", 0.01

    def probabilities(self, audio):
        n = int(len(audio) / SR / self.frame_sec)
        p = np.zeros((n, 4), np.float32)
        p[100:300, 0] = 0.9
        p[250:500, 2] = 0.8
        p[600:605, 0] = 0.7
        return p, 0.01

    def run(self, audio):
        p, t = self.probabilities(audio)
        stage = {"wall": t, "cpu": t, "cores": None}
        return p, {"features": stage, "queue_wait": 0.0, "inference": stage,
                   "accelerator": {"wall": 0.008, "calls": 2}, "cpu_side": {"wall": 0.002}}

    def info(self):
        return {"backend": self.backend, "accelerator": self.accelerator, "device": "cpu", "fallbacks": [],
                "requested_backend": "auto", "model": "fake", "frame_sec": self.frame_sec}


def wav_bytes(seconds=8.0):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * SR), np.float32), SR, format="WAV")
    return buf.getvalue()


class ApiTest(unittest.TestCase):
    def setUp(self):
        server.state.engine, server.state.status, server.state.error = FakeEngine(), "ready", None
        server.state.ready.set()
        settings.API_KEY = ""
        settings.MAX_UPLOAD_MB = 2048
        settings.MAX_AUDIO_MINUTES = 0
        self.client = TestClient(server.app)  # no `with`: the lifespan (real model load) does not run

    def post(self, path="/api/v1/diarize", data=None, content=None, headers=None):
        files = {"file": ("talk.wav", content if content is not None else wav_bytes(), "audio/wav")}
        return self.client.post(path, files=files, data=data or {}, headers=headers or {})

    def test_health_ready_info(self):
        self.assertEqual(self.client.get("/api/v1/health").json()["status"], "ok")
        self.assertEqual(self.client.get("/api/v1/ready").status_code, 200)
        info = self.client.get("/api/v1/info").json()
        self.assertEqual(info["backend"], "fake")
        self.assertFalse(info["auth_required"])

    def test_diarize_json(self):
        r = self.post()
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["num_speakers"], 2)  # the 50 ms blip is dropped by min_segment=0.1
        self.assertEqual([(s["speaker"], s["start"], s["end"]) for s in body["segments"]],
                         [("Speaker 1", 1.0, 3.0), ("Speaker 2", 2.5, 5.0)])
        self.assertEqual(body["segments"][0]["overlaps_with"], ["Speaker 2"])
        self.assertAlmostEqual(body["summary"]["overlap_time"], 0.5)
        self.assertAlmostEqual(body["summary"]["speech_time"], 4.0)
        self.assertEqual(body["summary"]["interruptions"], 1)
        self.assertEqual(body["speakers"][1]["interruptions"], 1)
        self.assertEqual(body["transitions"]["matrix"], [[0, 1], [0, 0]])
        self.assertNotIn("activity", body)
        t = body["timings"]
        self.assertTrue({"decode", "features", "inference", "accelerator", "cpu_side", "segmentation", "total"} <= set(t))
        self.assertEqual(t["total"], body["processing_time"])
        self.assertIn("accelerator 0.01s in 2 calls", server.format_timings(t))

    def test_parameters_and_probabilities(self):
        r = self.post(data={"min_segment": "0", "include_probabilities": "true"})
        body = r.json()
        self.assertEqual(len(body["segments"]), 3)
        self.assertEqual(body["parameters"]["min_segment"], 0)
        self.assertEqual(body["channels"], [0, 2])
        self.assertEqual(len(body["activity"]["probs"]), 800)

    def test_exports(self):
        for fmt, needle in [("rttm", "SPEAKER talk 1 1.000 2.000 <NA> <NA> Speaker_1"), ("csv", "Speaker 2,2.500,5.000,2.500"),
                            ("srt", "00:00:01,000 --> 00:00:03,000")]:
            r = self.post(data={"format": fmt})
            self.assertEqual(r.status_code, 200)
            self.assertIn(needle, r.text)
            self.assertIn(f'filename="talk.{fmt}"', r.headers["content-disposition"])

    def test_bad_requests(self):
        self.assertEqual(self.post(data={"threshold": "2"}).status_code, 422)
        self.assertEqual(self.post(data={"format": "xml"}).status_code, 422)
        self.assertEqual(self.post(content=b"").status_code, 400)
        self.assertEqual(self.post(content=b"this is not audio").status_code, 400)
        self.assertEqual(self.post(content=wav_bytes(0.2)).status_code, 400)

    def test_limits(self):
        settings.MAX_UPLOAD_MB = 1
        self.assertEqual(self.post(content=wav_bytes(40)).status_code, 413)  # 1.3 MB
        settings.MAX_UPLOAD_MB = 2048
        settings.MAX_AUDIO_MINUTES = 0.1
        self.assertEqual(self.post(content=wav_bytes(7)).status_code, 413)

    def test_not_ready(self):
        server.state.status = "loading"
        self.assertEqual(self.post().status_code, 503)
        self.assertEqual(self.client.get("/api/v1/ready").status_code, 503)
        server.state.status, server.state.error = "error", "boom"
        self.assertIn("boom", self.post().json()["detail"])

    def test_api_key(self):
        settings.API_KEY = "s3cret"
        self.assertEqual(self.post().status_code, 401)
        self.assertEqual(self.post(headers={"X-API-Key": "wrong"}).status_code, 401)
        self.assertEqual(self.post(headers={"X-API-Key": "s3cret"}).status_code, 200)
        self.assertEqual(self.post(headers={"Authorization": "Bearer s3cret"}).status_code, 200)
        self.assertEqual(self.client.get("/api/v1/jobs").status_code, 401)
        self.assertEqual(self.client.get("/api/v1/health").status_code, 200)  # probes stay public
        self.assertTrue(self.client.get("/api/v1/info").json()["auth_required"])

    def test_jobs(self):
        r = self.post("/api/v1/jobs", data={"threshold": "0.75"})
        self.assertEqual(r.status_code, 202, r.text)
        job = r.json()
        for _ in range(100):
            job = self.client.get(job["status_url"]).json()
            if job["status"] in ("done", "error"):
                break
            time.sleep(0.05)
        self.assertEqual(job["status"], "done", job)
        self.assertEqual(job["result"]["parameters"]["threshold"], 0.75)
        self.assertEqual(job["result"]["num_speakers"], 2)  # 0.9 and 0.8 pass 0.75; the 0.7 blip does not
        csv_text = self.client.get(job["result_url"], params={"format": "csv"}).text
        self.assertTrue(csv_text.startswith("speaker,start,end,duration"))
        self.assertIn(job["id"], [j["id"] for j in self.client.get("/api/v1/jobs").json()["jobs"]])
        self.assertEqual(self.client.delete(job["status_url"]).status_code, 204)
        self.assertEqual(self.client.get(job["status_url"]).status_code, 404)

    def test_legacy_routes(self):
        self.assertEqual(self.client.get("/api/info").json()["backend"], "fake")
        body = self.post("/api/diarize").json()
        self.assertIn("activity", body)
        self.assertIn("channels", body)


class SegmentationTest(unittest.TestCase):
    def test_merge_gap_and_order(self):
        p = np.zeros((1000, 3), np.float32)
        p[500:600, 2] = 1  # channel 2 speaks first ...
        p[620:700, 2] = 1  # ... with a 0.2 s gap (merged)
        p[800:900, 0] = 1
        r = analyze(p, 0.01, 10.0, merge_gap=0.3)
        self.assertEqual([(s["speaker"], s["start"], s["end"]) for s in r["segments"]],
                         [("Speaker 1", 5.0, 7.0), ("Speaker 2", 8.0, 9.0)])
        self.assertEqual(r["speakers"][0]["channel"], 2)
        r = analyze(p, 0.01, 10.0, merge_gap=0.1)
        self.assertEqual(len(r["segments"]), 3)

    def test_empty(self):
        r = analyze(np.zeros((500, 4), np.float32), 0.01, 5.0)
        self.assertEqual((r["num_speakers"], r["segments"], r["summary"]["speech_time"]), (0, [], 0))
        self.assertEqual(export({**r, "filename": "x.wav"}, "rttm")[0], "\n")


if __name__ == "__main__":
    unittest.main()
