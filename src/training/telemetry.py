"""Opt-in counters for actual fit calls, plus Windows process memory sampling."""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import ctypes
from ctypes import wintypes
import threading

_ACTIVE = ContextVar('fit_telemetry', default=None)


def track_fit(kind, X):
    active = _ACTIVE.get()
    if active is not None:
        active['counts'][kind] += 1
        active['shapes'][f'{kind}:{X.shape[0]}x{X.shape[1]}'] += 1


def progress(event):
    active = _ACTIVE.get()
    if active is not None and active['progress']:
        active['progress'](event)


@contextmanager
def fit_counts(callback=None):
    state = dict(counts=Counter(), shapes=Counter(), progress=callback)
    token = _ACTIVE.set(state)
    try:
        yield state
    finally:
        _ACTIVE.reset(token)


class MemorySampler:
    """50 ms sampled RSS/private bytes, plus OS lifetime high-water RSS. No global settings."""
    def __init__(self, interval=0.05):
        self.interval = interval
        self.stop = threading.Event()
        self.samples = []
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ['PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage']]
        self.Counters = Counters
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.psapi = ctypes.WinDLL('psapi', use_last_error=True)
        self.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        self.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    def snapshot(self):
        counters = self.Counters()
        counters.cb = ctypes.sizeof(counters)
        if not self.psapi.GetProcessMemoryInfo(self.kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return dict(rss=counters.WorkingSetSize / 2**20, private=counters.PrivateUsage / 2**20,
                    lifetime_peak_rss=counters.PeakWorkingSetSize / 2**20)

    def __enter__(self):
        self.samples.append(self.snapshot())
        def sample():
            while not self.stop.wait(self.interval):
                self.samples.append(self.snapshot())
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.samples.append(self.snapshot())

    def report(self):
        return dict(unit='MiB', sampling_interval_seconds=self.interval, samples=len(self.samples),
                    baseline_rss=self.samples[0]['rss'], end_rss=self.samples[-1]['rss'],
                    peak_sampled_rss=max(x['rss'] for x in self.samples),
                    peak_sampled_private=max(x['private'] for x in self.samples),
                    process_lifetime_peak_rss=self.samples[-1]['lifetime_peak_rss'],
                    scope='Fold computation including audit materialization; JSON serialization excluded. '
                          'OS peak covers lifetime of this isolated fold worker; sampled peak may miss brief spikes.')
