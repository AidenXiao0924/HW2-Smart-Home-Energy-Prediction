"""Download public real data, preserving provenance; no credentials needed/stored."""
import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

KAGGLE = 'https://www.kaggle.com/api/v1/datasets/download/loveall/appliances-energy-prediction'
UCI = 'https://archive.ics.uci.edu/static/public/374/appliances%2Benergy%2Bprediction.zip'


def acquire(root):
    root = Path(root)
    target = root / 'data/raw/energydata_complete.csv'
    if target.exists():
        raise FileExistsError('Refusing to overwrite original dataset')
    attempts = []
    for name, url in [('Kaggle', KAGGLE), ('UCI', UCI)]:
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'SmartHomeResearch/1.0'})
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read(30_000_001)
                final_url = response.url
            if len(payload) > 30_000_000:
                raise ValueError('Download exceeds size limit')
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                accepted = {target.name, 'KAG_energydata_complete.csv'}
                matches = [i for i in archive.infolist() if Path(i.filename).name in accepted]
                if len(matches) != 1 or matches[0].file_size > 30_000_000:
                    raise ValueError('Expected exactly one bounded energydata_complete.csv')
                csv_bytes = archive.read(matches[0])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(csv_bytes)
            metadata = dict(source=name, archive_member=matches[0].filename,
                            local_filename=target.name, requested_url=url, resolved_url=final_url,
                            downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
                            sha256=hashlib.sha256(csv_bytes).hexdigest(), bytes=len(csv_bytes),
                            archive_sha256=hashlib.sha256(payload).hexdigest(), attempts=attempts,
                            original_source='https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction',
                            license='UCI original: CC BY 4.0',
                            citation='Candanedo, L. (2017). Appliances Energy Prediction. DOI:10.24432/C5VC8G',
                            dataset_version='content-addressed by CSV SHA-256; Kaggle release number not asserted')
            evidence = root / 'reports/data'
            evidence.mkdir(parents=True, exist_ok=True)
            (evidence / 'provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            print(json.dumps(metadata, indent=2))
            return
        except Exception as exc:
            attempts.append(dict(source=name, url=url, error=f'{type(exc).__name__}: {exc}'))
    evidence = root / 'reports/data'
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / 'download_failures.json').write_text(json.dumps(attempts, indent=2), encoding='utf-8')
    raise RuntimeError(attempts)


if __name__ == '__main__':
    acquire(Path(__file__).resolve().parents[2])
