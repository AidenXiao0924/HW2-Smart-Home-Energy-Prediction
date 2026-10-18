"""Atomic, content-checked JSON checkpoints. Windows locks release on process exit."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import uuid
import msvcrt


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        stream.write(canonical(dict(payload=payload, sha256=digest(payload))))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_checkpoint(path):
    stored = json.loads(Path(path).read_text(encoding='utf-8'))
    if set(stored) != {'payload', 'sha256'} or digest(stored['payload']) != stored['sha256']:
        raise ValueError(f'Checkpoint integrity failure: {path}')
    return stored['payload']


@contextmanager
def run_lock(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'run.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('Another runner holds this run directory') from exc
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
