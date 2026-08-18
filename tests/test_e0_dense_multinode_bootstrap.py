import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "e0_dense_multinode_bootstrap.sh"


class DenseMultinodeBootstrapTests(unittest.TestCase):
    def test_bootstrap_requires_exact_four_by_eight_cluster(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('NNODES="${NNODES:-4}"', source)
        self.assertIn('NPROC_PER_NODE="${NPROC_PER_NODE:-8}"', source)
        self.assertIn('test "$NNODES" = 4', source)
        self.assertIn('test "$NPROC_PER_NODE" = 8', source)

    def test_bootstrap_requires_multinode_coordinates(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("NODE_RANK:?", source)
        self.assertIn("MASTER_ADDR:?", source)
        self.assertIn('--rdzv-endpoint "$MASTER_ADDR:$MASTER_PORT"', source)

    def test_only_head_node_hashes_final_evidence(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('if [[ "$NODE_RANK" = 0', source)
        self.assertIn("sha256sum", source)
        self.assertIn('exit "$validation_status"', source)


if __name__ == "__main__":
    unittest.main()
