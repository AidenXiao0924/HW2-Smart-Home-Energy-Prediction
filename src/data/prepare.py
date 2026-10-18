import hashlib
import json
from pathlib import Path
from src.data.validate import read_raw, quality_report
from src.features.engineering import engineer, CANDIDATE_FEATURES, DEPLOYABLE_FEATURES
from src.data.split import chronological_split, csv_bytes


def prepare(root):
    root = Path(root)
    raw = root / 'data/raw/energydata_complete.csv'
    frame = read_raw(raw)
    quality = quality_report(frame)
    quality['raw_sha256'] = hashlib.sha256(raw.read_bytes()).hexdigest()
    quality['candidate_features'] = CANDIDATE_FEATURES
    quality['candidate_count'] = len(CANDIDATE_FEATURES)
    quality['deployable_count'] = len(DEPLOYABLE_FEATURES)
    parts, manifest = chronological_split(engineer(frame))
    manifest['raw_sha256'] = quality['raw_sha256']
    manifest['time_semantics'] = 'Source-local naive calendar; Belgium source; no timezone conversion'
    out = root / 'reports/data'
    out.mkdir(parents=True, exist_ok=True)
    processed = root / 'data/processed'
    processed.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        (processed / f'{name}.csv').write_bytes(csv_bytes(part))
    for name, payload in [('quality_report', quality), ('split_manifest', manifest)]:
        (out / f'{name}.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    prepare(Path(__file__).resolve().parents[2])
