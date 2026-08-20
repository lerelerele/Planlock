import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "e0-nccl-pe-dense-evidence.json"
MANIFEST = ROOT / "e0-manifest-candidate.json"


def test_physical_dense_evidence_validates_both_pes_without_closing_e0() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    canonical = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest_digest = hashlib.sha256(canonical).hexdigest()

    assert evidence["status"] == "CONFIRMED_PHYSICAL_NCCL_PE_DENSE"
    assert evidence["manifest_sha256"] == manifest_digest
    assert evidence["reference_sha"] == manifest["reference_sha"]
    assert evidence["source_artifact"]["storage"] == "external"
    assert len(evidence["source_artifact"]["sha256"]) == 64
    assert evidence["nccl_probe"] == {
        "world_size": 32,
        "unique_gpu_uuids": 32,
        "distinct_hostnames": 8,
        "rank_values": "1..32",
        "expected_all_reduce_sum": 528,
        "observed_all_reduce_sum": 528,
        "completed": True,
    }
    assert evidence["training"]["mesh_built"] is True
    assert evidence["training"]["trainer_step_completed"] is True
    assert evidence["claims"] == {
        "pe_moe_physical_nccl_validated": True,
        "pe_dense_physical_nccl_validated": True,
        "both_pes_physical_nccl_validated": True,
        "e0_closed": False,
    }
