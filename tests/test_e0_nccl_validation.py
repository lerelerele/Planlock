import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "e0_nccl_validation.py"
SPEC = importlib.util.spec_from_file_location("e0_nccl_validation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NcclValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (ROOT / "e0-manifest-candidate.json").read_text(encoding="utf-8")
        )

    def test_manifest_freezes_eight_rank_standard_moe(self) -> None:
        pe = MODULE.validate_manifest(self.manifest)
        self.assertEqual(MODULE.world_size(pe), 8)
        self.assertEqual(pe["overrides"]["moe_comm_backend"], "standard")

    def test_generated_runtime_config_compiles(self) -> None:
        source = MODULE.runtime_module(self.manifest["pes"]["PE_moe"])
        compile(source, "<planlock_e0_runtime>", "exec")
        self.assertIn("config.parallelism.expert_parallel_degree = 2", source)
        self.assertIn("config.parallelism.pipeline_parallel_degree = 2", source)

    def test_torchrun_uses_exactly_eight_local_processes(self) -> None:
        command = MODULE.torchrun_command("probe")
        self.assertIn("--nproc_per_node=8", command)
        self.assertIn("--rdzv_endpoint=localhost:0", command)

    def test_nccl_probe_requires_expected_all_reduce_sum(self) -> None:
        source = MODULE.nccl_probe_source()
        self.assertIn("dist.init_process_group(\"nccl\")", source)
        self.assertIn("dist.all_reduce(value)", source)
        self.assertIn("PLANLOCK_NCCL_PROBE=", source)

    @patch.object(MODULE, "run_training")
    @patch.object(MODULE, "run_nccl_probe")
    @patch.object(MODULE, "nvidia_topology", return_value={"inventory": "8", "topology": "PCIe"})
    @patch.object(
        MODULE,
        "cuda_probe",
        return_value={"cuda_available": True, "device_count": 8, "nccl_available": True},
    )
    @patch.object(MODULE, "git_head")
    def test_confirmed_report_still_keeps_e0_open(
        self, git_head, _cuda, _topology, nccl, training
    ) -> None:
        git_head.side_effect = [MODULE.REFERENCE_SHA, "planlock-sha"]
        nccl.return_value = {"status": "CONFIRMED_NCCL_ALL_REDUCE"}
        training.return_value = {
            "status": "CONFIRMED_PHYSICAL_NCCL_PE_MOE",
            "mesh_built": True,
            "training_completed": True,
        }
        report = MODULE.run(Path("planlock"), Path("torchtitan"), self.manifest)
        self.assertEqual(report["status"], "CONFIRMED_PHYSICAL_NCCL_PE_MOE")
        self.assertTrue(report["claims"]["pe_moe_physical_nccl_validated"])
        self.assertFalse(report["claims"]["pe_dense_physical_nccl_validated"])
        self.assertFalse(report["claims"]["e0_closed"])

    @patch.object(MODULE, "run_training")
    @patch.object(MODULE, "run_nccl_probe", return_value={"status": "FAILED_NCCL_PROBE"})
    @patch.object(MODULE, "nvidia_topology", return_value={})
    @patch.object(MODULE, "cuda_probe", return_value={})
    @patch.object(MODULE, "git_head")
    def test_training_is_skipped_after_nccl_failure(
        self, git_head, _cuda, _topology, _nccl, training
    ) -> None:
        git_head.side_effect = [MODULE.REFERENCE_SHA, "planlock-sha"]
        report = MODULE.run(Path("planlock"), Path("torchtitan"), self.manifest)
        training.assert_not_called()
        self.assertEqual(report["training"]["status"], "SKIPPED_AFTER_NCCL_FAILURE")
        self.assertEqual(report["status"], "FAILED_PHYSICAL_NCCL_PE_MOE")


if __name__ == "__main__":
    unittest.main()
