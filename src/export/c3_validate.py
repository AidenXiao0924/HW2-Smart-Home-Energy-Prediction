"""Read-only deployment bundle validation, requiring no dataset or Python model."""
import json
from src.export.c3 import PUBLIC,PRIVATE,ROOT,load_json
from src.export.c3_contracts import validate,public_safe
from src.export.reference import validate_artifact
from src.training.c1 import C1Gate
from src.training.b3_io import sha
from src.training.checkpoint import read_checkpoint


def verify():
    with C1Gate('none') as gate:
        manifest=load_json(PUBLIC/'manifest.json')
        for path,record in manifest['files'].items():
            if sha(PUBLIC/path)!=record['sha256'] or (PUBLIC/path).stat().st_size!=record['bytes']:
                raise ValueError('Public artifact hash/size mismatch')
        for path in PUBLIC.rglob('*.json'):public_safe(load_json(path))
        payloads={'model':'model.json','research-summary':'research_summary.json','parity-fixtures':'parity_fixtures.json',
            'test-series':'charts/test_series.json','test-residuals':'charts/test_residuals.json','training-cv':'charts/training_cv.json'}
        for name,path in payloads.items():validate(load_json(PUBLIC/path),load_json(PUBLIC/f'schemas/{name}.schema.json'))
        validate_artifact(load_json(PUBLIC/'model.json'))
        consolidated=read_checkpoint(PRIVATE/'consolidation_manifest.json')
        for group in ['sources','outputs','source_code']:
            for path,value in consolidated[group].items():
                if sha(ROOT/path)!=value:raise ValueError('Research provenance mismatch')
        return dict(status='PASS',validated_payload_types=len(payloads),schema_files=len(list((PUBLIC/'schemas').glob('*.json'))),
            public_json_files=len(list(PUBLIC.rglob('*.json'))),actual_public_bytes=sum(p.stat().st_size for p in PUBLIC.rglob('*') if p.is_file()),
            homepage_bytes=manifest['budgets']['homepage_json_bytes'],source_dataset_reads=gate.reads,
            model_prediction_calls=0,model_version=manifest['model_version'])


if __name__=='__main__':print(json.dumps(verify(),indent=2))
