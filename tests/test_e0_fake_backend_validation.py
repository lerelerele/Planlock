import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).parents[1] / "scripts" / "e0_fake_backend_validation.py"
SPEC = importlib.util.spec_from_file_location("e0_fake_backend_validation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeBackendValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pe = {
            "modulo_registro": "torchtitan.models.llama3.config_registry",
            "funcion_config": "llama3_debugmodel",
            "overrides": {
                "data_parallel_replicate_degree": 2,
                "data_parallel_shard_degree": 2,
                "context_parallel_degree": 2,
                "tensor_parallel_degree": 2,
                "pipeline_parallel_degree": 2,
                "expert_parallel_degree": 1,
                "enable_sequence_parallel": True,
                "spmd_backend": "eager",
                "pipeline_parallel_schedule": "Interleaved1F1B",
                "module_fqns_per_model_part": [["tok_embeddings"], ["lm_head"]],
            },
        }

    def test_derives_dense_fake_world_size(self) -> None:
        self.assertEqual(MODULE.world_size(self.pe), 32)

    def test_runtime_module_applies_manifest_overrides(self) -> None:
        source = MODULE.runtime_module("PE_dense", self.pe)
        self.assertIn("from torchtitan.models.llama3.config_registry", source)
        self.assertIn("config.parallelism.tensor_parallel_degree = 2", source)
        self.assertIn("def PE_dense():", source)

    @patch.object(MODULE.subprocess, "run")
    def test_runner_disables_cuda_graphs_for_pipeline_parallelism(self, run) -> None:
        run.return_value.returncode = 0
        run.return_value.stdout = (
            "Building device mesh with parallelism:\nTraining completed\n"
        )
        run.return_value.stderr = ""
        MODULE.run_candidate(Path("repo"), "PE_dense", self.pe, 60)
        command = run.call_args.args[0]
        self.assertIn("--training.disable_cuda_graphs", command)

    def test_rejects_output_inside_checkout(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            MODULE.external_output(SCRIPT.parent / "report.json")
        with tempfile.TemporaryDirectory() as directory:
            expected = (Path(directory) / "report.json").resolve()
            self.assertEqual(MODULE.external_output(expected), expected)

    @patch.object(MODULE, "git_state", return_value="wrong")
    def test_fails_closed_on_reference_sha_mismatch(self, _git_state) -> None:
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            MODULE.run(Path("repo"), {"reference_sha": MODULE.REFERENCE_SHA, "pes": {}})

    @patch.object(MODULE, "run_candidate")
    @patch.object(
        MODULE,
        "cuda_probe",
        return_value={"cuda_available": True, "device_count": 1, "device_name": "GPU"},
    )
    @patch.object(MODULE, "git_state", return_value=MODULE.REFERENCE_SHA)
    def test_report_cannot_claim_nccl_or_e0_closure(
        self, _git_state, _cuda_probe, run_candidate
    ) -> None:
        run_candidate.return_value = {
            "status": "CONFIRMED_CUDA_FAKE_BACKEND",
            "world_size": 32,
        }
        manifest = {
            "reference_sha": MODULE.REFERENCE_SHA,
            "pes": {"PE_dense": self.pe},
        }
        report = MODULE.run(Path("repo"), manifest, ("PE_dense",))
        self.assertFalse(report["nccl_collectives_validated"])
        self.assertFalse(report["physical_multi_gpu_validated"])
        self.assertFalse(report["e0_closed"])


if __name__ == "__main__":
    unittest.main()
