import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "PREREG_V14_DIGESTS.txt"
TRACKED = {
    "preregistro-huella-estructural-v14.md",
    "e0-nccl-pe-dense-evidence.json",
    "e0-manifest-candidate.json",
}


def canonical_repository_bytes(path: Path) -> bytes:
    text = path.read_text(encoding="utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def test_prereg_digest_record_matches_canonical_repository_bytes() -> None:
    entries = {
        match.group("name"): match.group("digest")
        for match in re.finditer(
            r"^(?P<digest>[0-9a-f]{64})  (?P<name>\S+)$",
            RECORD.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        if match.group("name") in TRACKED
    }

    assert entries.keys() == TRACKED
    for name, expected in entries.items():
        observed = hashlib.sha256(canonical_repository_bytes(ROOT / name)).hexdigest()
        assert observed == expected, f"digest drift for {name}"
