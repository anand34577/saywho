"""Per-stage timing for the processing breakdown (`timings` in API results, one log line per file)."""
import os
import time


def _cpu_seconds() -> float:
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


class stage_clock:
    """Measures one stage: wall seconds, and CPU seconds of this process plus finished child processes (ffmpeg).

    cpu / wall = average number of busy cores during the stage. Approximate when several requests run at once.
    Child-process CPU time (ffmpeg) is only counted on Linux/macOS.
    """

    def __init__(self):
        self.wall, self.cpu = time.perf_counter(), _cpu_seconds()

    def stop(self) -> dict:
        wall = time.perf_counter() - self.wall
        cpu = _cpu_seconds() - self.cpu
        return {"wall": round(wall, 3), "cpu": round(cpu, 3), "cores": round(cpu / wall, 1) if wall > 0.01 else None}
