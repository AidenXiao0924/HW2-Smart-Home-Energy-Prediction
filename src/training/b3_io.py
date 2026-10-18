"""Immutable B3 records and hard phase-specific dataset read boundaries."""
import builtins
from datetime import datetime, timezone
import hashlib
import io
import os
from pathlib import Path
from src.training.checkpoint import canonical, digest, read_checkpoint


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, payload):
    """Exclusive creation: no versioned record can silently replace an earlier one."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical(dict(payload=payload,sha256=digest(payload))))
        stream.flush()
        os.fsync(stream.fileno())


class DatasetGate:
    def __init__(self, root, phase):
        if phase not in ['lock','train','validation']:
            raise ValueError('Unknown dataset access phase')
        self.root=Path(root).resolve()
        self.phase=phase
        self.reads=[]
        self.allowed=set()
        if phase in ['train','validation']: self.allowed.add(self.root/'data/processed/train.csv')
        if phase=='validation': self.allowed.add(self.root/'data/processed/validation.csv')

    def __enter__(self):
        self.old_open,self.old_io_open=builtins.open,io.open
        def guard(original):
            def opened(file,mode='r',*args,**kwargs):
                if not isinstance(file,int):
                    path=Path(os.fspath(file)).resolve()
                    if path.is_relative_to(self.root/'data'):
                        if path not in self.allowed or any(c in mode for c in 'wax+'):
                            raise PermissionError(f'B3 {self.phase} blocks data access: {path}')
                        self.reads.append(dict(at_utc=now(),phase=self.phase,path=str(path.relative_to(self.root)).replace('\\','/'),mode=mode))
                return original(file,mode,*args,**kwargs)
            return opened
        builtins.open=guard(self.old_open)
        io.open=guard(self.old_io_open)
        return self

    def __exit__(self,*args):
        builtins.open,self.old_open=self.old_open,None
        io.open,self.old_io_open=self.old_io_open,None


def protected_validation_load(root, folder, loader):
    """A Validation CSV loader cannot execute before immutable lock + training manifest + ledger exist."""
    folder=Path(folder)
    lock=read_checkpoint(folder/'candidate_lock.json')
    trained=read_checkpoint(folder/'training_manifest.json')
    ledger=read_checkpoint(folder/'validation_access.json')
    if (trained['status']!='PASS' or trained['lock_sha256']!=sha(folder/'candidate_lock.json')
        or ledger['training_manifest_sha256']!=sha(folder/'training_manifest.json')
        or ledger['lock_sha256']!=sha(folder/'candidate_lock.json')
        or len(lock['candidates'])*3!=len(trained['models'])):
        raise ValueError('Validation preconditions or immutable provenance mismatch')
    return loader()
