import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "e0_dense_multinode_validation.py"
SPEC = importlib.util.spec_from_file_location("e0_dense_multinode_validation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DenseMultinodeValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (ROOT / "e0-manifest-candidate.json").read_text(encoding="utf-8")
        )
        cls.moe_evidence = json.loads(
            (ROOT / "e0-nccl-pe-moe-evidence.json").read_text(encoding="utf-8")
        )

    def test_manifest_freezes_thirty_two_rank_dense_candidate(self) -> None:
        pe = MODULE.validate_manifest(self.manifest)
        self.assertEqual(MODULE.world_size(pe), 32)
        self.assertEqual(pe["overrides"]["pipeline_parallel_schedule"], "Interleaved1F1B")

    def test_prior_moe_evidence_is_bound_to_same_manifest(self) -> None:
        digest = MODULE.hashlib.sha256(MODULE.canonical_bytes(self.manifest)).hexdigest()
        prior = MODULE.validate_pe_moe_evidence(self.moe_evidence, digest)
        self.assertEqual(prior["status"], "CONFIRMED_PHYSICAL_NCCL_PE_MOE")

    def test_generated_runtime_config_compiles(self) -> None:
        source = MODULE.runtime_module(self.manifest["pes"]["PE_dense"])
        compile(source, "<planlock_e0_dense_runtime>", "exec")
        self.assertIn("config.parallelism.context_parallel_degree = 2", source)
        self.assertIn("config.parallelism.pipeline_parallel_degree = 2", source)

    def test_torchrun_uses_requested_uniform_layout(self) -> None:
        command = MODULE.torchrun_command(
            "probe",
            nnodes=4,
            node_rank=2,
            nproc_per_node=8,
            rdzv_endpoint="10.0.0.1:29500",
            rdzv_id="probe-id",
            local_address="10.0.0.3",
        )
        self.assertIn("--nnodes=4", command)
        self.assertIn("--node_rank=2", command)
        self.assertIn("--nproc_per_node=8", command)
        self.assertIn("--master_addr=10.0.0.1", command)
        self.assertIn("--master_port=29500", command)
        self.assertNotIn("--rdzv_backend=c10d", command)
        self.assertIn("--local_addr=10.0.0.3", command)

        head_command = MODULE.torchrun_command(
            "probe",
            nnodes=4,
            node_rank=0,
            nproc_per_node=8,
            rdzv_endpoint="10.0.0.1:29500",
            rdzv_id="probe-id",
            local_address="10.0.0.1",
        )
        self.assertIn("--node_rank=0", head_command)

    def test_rank_inventory_requires_unique_gpus_on_uniform_distinct_hosts(self) -> None:
        inventory = [
            {
                "rank": rank,
                "hostname": f"node-{rank // 4}",
                "device_uuid": f"GPU-{rank}",
            }
            for rank in range(32)
        ]
        self.assertTrue(
            MODULE.validate_rank_inventory(inventory, nnodes=8, nproc_per_node=4)
        )
        inventory[-1]["device_uuid"] = inventory[0]["device_uuid"]
        self.assertFalse(
            MODULE.validate_rank_inventory(inventory, nnodes=8, nproc_per_node=4)
        )

    def test_nccl_probe_requires_32_rank_sum_and_inventory(self) -> None:
        source = MODULE.nccl_probe_source()
        self.assertIn('dist.init_process_group("nccl")', source)
        self.assertIn("dist.all_gather_object(identities, identity)", source)
        self.assertIn("dist.all_reduce(value)", source)
        self.assertIn("PLANLOCK_NCCL_PROBE_FILE", source)
        self.assertIn("Path(probe_file).write_text", source)
        self.assertEqual(MODULE.EXPECTED_ALL_REDUCE_SUM, 528)

    def test_localhost_rendezvous_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "head node address"):
            MODULE.run(
                Path("planlock"),
                Path("torchtitan"),
                self.manifest,
                self.moe_evidence,
                nnodes=4,
                node_rank=0,
                nproc_per_node=8,
                rdzv_endpoint="localhost:29500",
            )

    def test_single_node_and_non_32_gpu_layouts_are_rejected(self) -> None:
        for nnodes, nproc in ((1, 32), (8, 2), (3, 8)):
            with (
                self.subTest(nnodes=nnodes, nproc=nproc),
                self.assertRaisesRegex(ValueError, "at least two|32 GPUs"),
            ):
                MODULE.run(
                    Path("planlock"),
                    Path("torchtitan"),
                    self.manifest,
                    self.moe_evidence,
                    nnodes=nnodes,
                    node_rank=0,
                    nproc_per_node=nproc,
                    rdzv_endpoint="10.0.0.1:29500",
                )

    @patch.object(MODULE, "run_training")
    @patch.object(MODULE, "run_nccl_probe")
    @patch.object(MODULE, "local_nvidia_topology", return_value={"inventory": "8", "topology": "NVLink"})
    @patch.object(
        MODULE,
        "cuda_probe",
        return_value={"cuda_available": True, "device_count": 8, "nccl_available": True},
    )
    @patch.object(MODULE, "git_head")
    def test_dense_confirmation_closes_e0_with_prior_moe_evidence(
        self, git_head, _cuda, _topology, nccl, training
    ) -> None:
        git_head.side_effect = [MODULE.REFERENCE_SHA, "planlock-sha"]
        nccl.return_value = {"status": "CONFIRMED_NCCL_ALL_REDUCE"}
        training.return_value = {
            "status": "CONFIRMED_PHYSICAL_NCCL_PE_DENSE",
            "mesh_built": True,
            "training_completed": True,
        }
        report = MODULE.run(
            Path("planlock"),
            Path("torchtitan"),
            self.manifest,
            self.moe_evidence,
            nnodes=4,
            node_rank=0,
            nproc_per_node=8,
            rdzv_endpoint="10.0.0.1:29500",
        )
        self.assertEqual(report["status"], "CONFIRMED_PHYSICAL_NCCL_PE_DENSE")
        self.assertEqual(report["cluster"]["world_size"], 32)
        self.assertTrue(report["claims"]["pe_dense_physical_nccl_validated"])
        self.assertTrue(report["claims"]["pe_moe_physical_nccl_validated"])
        self.assertTrue(report["claims"]["e0_closed"])

    @patch.object(MODULE, "run_training")
    @patch.object(MODULE, "run_nccl_probe")
    @patch.object(MODULE, "local_nvidia_topology", return_value={"inventory": "4", "topology": "PIX"})
    @patch.object(
        MODULE,
        "cuda_probe",
        return_value={"cuda_available": True, "device_count": 4, "nccl_available": True},
    )
    @patch.object(MODULE, "git_head")
    def test_eight_by_four_layout_can_close_e0(
        self, git_head, _cuda, _topology, nccl, training
    ) -> None:
        git_head.side_effect = [MODULE.REFERENCE_SHA, "planlock-sha"]
        nccl.return_value = {"status": "CONFIRMED_NCCL_ALL_REDUCE"}
        training.return_value = {
            "status": "CONFIRMED_PHYSICAL_NCCL_PE_DENSE",
            "mesh_built": True,
            "training_completed": True,
        }
        report = MODULE.run(
            Path("planlock"),
            Path("torchtitan"),
            self.manifest,
            self.moe_evidence,
            nnodes=8,
            node_rank=0,
            nproc_per_node=4,
            rdzv_endpoint="10.0.0.1:29500",
        )
        self.assertEqual(report["cluster"]["nnodes"], 8)
        self.assertEqual(report["cluster"]["nproc_per_node"], 4)
        self.assertTrue(report["claims"]["e0_closed"])


if __name__ == "__main__":
    unittest.main()
