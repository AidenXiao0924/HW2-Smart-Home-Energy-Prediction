import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_release_gate_is_technical_not_authorization_or_submission():
    gate = load("reports/release/m4a3/RELEASE_GATE_DECISION.json")
    assert gate["TECHNICAL_RELEASE_READY"] is True
    assert gate["AUTHORIZED_TO_DEPLOY"] is False
    assert gate["PUBLIC_DEPLOYMENT_GATE"] is False
    assert gate["CAN_ENTER_M4B"] is False
    assert gate["ASSIGNMENT_READY"] is False
    assert gate["public_deployment_performed"] is False
    assert gate["final_zip_created"] is False


def test_traceability_only_closes_evidenced_root_gaps():
    trace = load("docs/technical/PDD_REQUIREMENT_TRACEABILITY.json")
    assert trace["counts_all"] == {"COMPLETE": 28, "PARTIAL": 8, "NOT_IMPLEMENTED": 4, "NOT_APPLICABLE": 2}
    rows = {r["id"]: r for r in trace["requirements"]}
    assert rows["UI-DISTRIBUTION"]["status"] == "COMPLETE"
    assert rows["UI-OOD"]["status"] == "COMPLETE"
    assert rows["UI-OVERVIEW"]["status"] == "PARTIAL"
    assert rows["UI-METRIC"]["status"] == "PARTIAL"
    assert rows["REQ-024"]["status"] == "NOT_IMPLEMENTED"
    assert rows["DEP-PAGES"]["status"] == "NOT_IMPLEMENTED"


def test_acceptance_manifest_matches_rc3_and_frozen_hashes():
    evidence = load("reports/web/m4a3/acceptance_manifest.json")
    rc = load("releases/web-rc3/release-manifest.json")
    assert evidence["tests"]["node"] == 182
    assert evidence["tests"]["pytest"] == 64
    assert evidence["tests"]["parity"] == 23
    assert evidence["tests"]["playwright_total"] == 154
    assert evidence["release_candidate"]["tree_sha256"] == rc["tree_sha256"]
    assert evidence["release_candidate"]["manifest_sha256"] == sha("releases/web-rc3/release-manifest.json")
    assert evidence["frozen_integrity"]["private_model"] == "8ab483ebb9044373009c070b179c365ce31c53a9a2a4378230bebbbb7326f08d"
    assert evidence["frozen_integrity"]["public_model"] == "da35fb7752cc356efa5dc8e79d6cdc63785ca8421aca344385a16a4c8f520e28"
    assert evidence["conformal_q_wh"] == 185.3571214361994


def test_rc1_rc2_and_pdd_remain_unchanged():
    assert sha("docs/pdd/Final_PDD_v1.0.md") == "0f08a6a01485da26d914746240da45f09d2b3198f52a646627142a97b36e87ef"
    assert sha("releases/web-rc1/release-manifest.json") == "69409b209f289e2cc52d9f32c578cd4fb05efed7b6c654832684c8d06f82c04a"
    assert sha("releases/web-rc2/release-manifest.json") == "912638af1deef4489c650fe4a5d7e54d3d84ae6c4fc53d6aec2890fc150b8a3f"


def test_public_training_package_is_aggregate_only_and_identical():
    allowed = {"manifest.json", "ranges.json", "schema.json", "summary.json"}
    public = ROOT / "releases/web-rc3/site/public/training-v1"
    canonical = ROOT / "artifacts/training/train-v1"
    assert {p.name for p in public.iterdir() if p.is_file()} == allowed
    for name in allowed:
        assert (public / name).read_bytes() == (canonical / name).read_bytes()
    sensitive = [p for p in (ROOT / "releases/web-rc3/site").rglob("*") if p.suffix.lower() in {".csv", ".joblib", ".pkl"}]
    assert [p.relative_to(ROOT / "releases/web-rc3/site").as_posix() for p in sensitive] == ["public/test-demo-v1/demo.csv"]
