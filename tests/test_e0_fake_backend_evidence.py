import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
EVIDENCE = ROOT / "e0-cuda-fake-backend-evidence.json"
MANIFEST = ROOT / "e0-manifest-candidate.json"


class FakeBackendEvidenceTests(unittest.TestCase):
    def test_evidence_matches_candidate_manifest_and_stays_open(self) -> None:
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        canonical = json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.assertEqual(
            evidence["manifest_sha256"], hashlib.sha256(canonical).hexdigest()
        )
        self.assertEqual(evidence["reference_sha"], manifest["reference_sha"])
        self.assertEqual(set(evidence["pes"]), {"PE_dense", "PE_moe"})
        self.assertTrue(all(row["mesh_built"] for row in evidence["pes"].values()))
        self.assertTrue(
            all(not row["training_completed"] for row in evidence["pes"].values())
        )
        self.assertFalse(evidence["claims"]["nccl_collectives_validated"])
        self.assertFalse(evidence["claims"]["physical_multi_gpu_validated"])
        self.assertFalse(evidence["claims"]["e0_closed"])

    def test_external_artifact_has_full_sha256(self) -> None:
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        digest = evidence["source_artifact"]["sha256"]
        self.assertEqual(len(digest), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in digest))


if __name__ == "__main__":
    unittest.main()
