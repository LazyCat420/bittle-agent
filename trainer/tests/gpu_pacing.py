"""Cooperative gaps between complete GPU work units; preserves their results.

The duty setting is a submission target, not a driver utilization cap. A single
kernel can still be busy. Full-speed mode adds no synchronization or sleeps.
"""
import time


class GpuPacer:
    def __init__(self, percent, wait, clock=time.monotonic, sleep=time.sleep):
        if not 10 <= percent <= 100:
            raise ValueError('GPU duty must be between 10 and 100 percent')
        self.percent, self.wait, self.clock, self.sleep = percent, wait, clock, sleep
        self.calls = 0
        self.work_seconds = self.idle_seconds = 0.0
        self.clamped_calls = 0

    def run(self, function, *args, **kwargs):
        if self.percent == 100:
            return function(*args, **kwargs)
        started = self.clock()
        value = function(*args, **kwargs)
        self.wait(value)  # Drain this work unit before giving other GPU users time.
        work = self.clock() - started
        requested = work * (100 / self.percent - 1)
        idle = min(requested, 0.25)  # Cold CPU compilation must not create long gaps.
        self.sleep(idle)
        self.calls += 1
        self.work_seconds += work
        self.idle_seconds += idle
        self.clamped_calls += requested > idle
        return value

    def stats(self):
        return {'duty_percent': self.percent, 'paced_calls': self.calls,
                'work_wall_seconds': self.work_seconds, 'idle_seconds': self.idle_seconds,
                'clamped_calls': self.clamped_calls}
