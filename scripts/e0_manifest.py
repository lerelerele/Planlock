#!/usr/bin/env python3
"""Validate and hash the candidate E0 PE manifest without importing TorchTitan."""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REFERENCE_SHA = "9a711521ac2973fe230a3f38efc6aedfc7d1f9c6"
EXPECTED_WORLD_SIZE = {"PE_dense": 32, "PE_moe": 8}
EXPECTED_OPTIMIZER = {
    "name": "AdamW",
    "implementation": "fused",
    "amsgrad": False,
    "state_tensors": {
        "exp_avg": "same_as_param",
        "exp_avg_sq": "same_as_param",
        "step": "f32",
    },
}


def canonical_bytes(manifest: dict[str, object]) -> bytes:
    return json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def config_uses_default_adamw(path: Path, function_name: str) -> bool:
    """Check the selected registry function's Trainer.Config optimizer keyword."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name),
        None,
    )
    if function is None:
        return False
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "optimizer" or not isinstance(keyword.value, ast.Call):
                continue
            callee = keyword.value.func
            return isinstance(callee, ast.Name) and callee.id == "default_adamw"
    return False


def default_adamw_contract(path: Path) -> dict[str, object]:
    """Extract the reviewed optimizer name and container implementation default."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    implementation = None
    optimizer_name = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "implementation"
            and isinstance(node.value, ast.Constant)
        ):
            implementation = node.value.value
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "default_adamw"),
        None,
    )
    if function is not None:
        for node in ast.walk(function):
            if not isinstance(node, ast.keyword) or node.arg != "optimizer_name":
                continue
            if isinstance(node.value, ast.Constant):
                optimizer_name = node.value.value
    return {"name": optimizer_name, "implementation": implementation}


def function_parameter_default(path: Path, function_name: str, parameter: str) -> object:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name),
        None,
    )
    if function is None:
        return None
    positional = [*function.args.posonlyargs, *function.args.args]
    positional_defaults = [None] * (len(positional) - len(function.args.defaults)) + list(function.args.defaults)
    pairs = list(zip(positional, positional_defaults, strict=True)) + list(
        zip(function.args.kwonlyargs, function.args.kw_defaults, strict=True)
    )
    for argument, default in pairs:
        if argument.arg == parameter and isinstance(default, ast.Constant):
            return default.value
    return None


def function_keyword_constant(path: Path, function_name: str, keyword_name: str) -> object:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name),
        None,
    )
    if function is None:
        return None
    values = [
        keyword.value.value
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == keyword_name and isinstance(keyword.value, ast.Constant)
    ]
    return values[0] if len(values) == 1 else None


def validate_partition(parts: list[list[str]], layer_count: int = 6) -> None:
    flattened = [name for part in parts for name in part]
    expected = ["tok_embeddings", *(f"layers.{i}" for i in range(layer_count)), "norm", "lm_head"]
    if sorted(flattened) != sorted(expected):
        raise ValueError("pipeline partition must cover each debugmodel module exactly once")
    if any(not part for part in parts):
        raise ValueError("pipeline partition contains an empty model part")


def validate(repo: Path, manifest: dict[str, object]) -> dict[str, object]:
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != REFERENCE_SHA or manifest.get("reference_sha") != actual:
        raise ValueError("reference SHA mismatch")
    if manifest.get("status") != "CANDIDATE_NOT_FROZEN":
        raise ValueError("this validator only accepts an unfrozen candidate manifest")
    optimizer_contract = default_adamw_contract(
        repo / "torchtitan/components/optimizer.py"
    )
    if optimizer_contract != {"name": "AdamW", "implementation": "fused"}:
        raise ValueError(f"unexpected default AdamW contract: {optimizer_contract}")
    dispatcher_default = function_parameter_default(
        repo / "torchtitan/models/deepseek_v3/__init__.py",
        "model_registry",
        "moe_comm_backend",
    )
    if dispatcher_default != "standard":
        raise ValueError(f"unexpected DeepSeek debugmodel dispatcher: {dispatcher_default}")
    for model_name in ("llama3", "deepseek_v3"):
        attention_default = function_parameter_default(
            repo / f"torchtitan/models/{model_name}/__init__.py",
            "model_registry",
            "attn_backend",
        )
        if attention_default != "flex":
            raise ValueError(
                f"unexpected {model_name} attention backend: {attention_default}"
            )
    fused_qkv = function_keyword_constant(
        repo / "torchtitan/models/llama3/__init__.py", "_debugmodel", "fuse_qkv"
    )
    if fused_qkv is not True:
        raise ValueError(f"unexpected Llama debugmodel fuse_qkv: {fused_qkv}")

    results = {}
    pes = manifest.get("pes")
    if not isinstance(pes, dict) or set(pes) != set(EXPECTED_WORLD_SIZE):
        raise ValueError("manifest must contain exactly PE_dense and PE_moe")
    for pe_name, expected_world_size in EXPECTED_WORLD_SIZE.items():
        pe = pes[pe_name]
        module_path = repo / (str(pe["modulo_registro"]).replace(".", "/") + ".py")
        if not module_path.is_file():
            raise ValueError(f"missing config registry for {pe_name}: {module_path}")
        if pe["funcion_config"] not in function_names(module_path):
            raise ValueError(f"missing function_config for {pe_name}: {pe['funcion_config']}")
        if not config_uses_default_adamw(module_path, pe["funcion_config"]):
            raise ValueError(f"{pe_name} selected config does not use default_adamw")
        overrides = pe["overrides"]
        architecture = pe["arquitectura"]
        if pe_name == "PE_moe" and overrides.get("moe_comm_backend") != "standard":
            raise ValueError("PE_moe must freeze the reviewed standard dispatcher")
        if overrides.get("attn_backend") != "flex":
            raise ValueError(f"{pe_name} must freeze the reviewed flex attention backend")
        if pe.get("dtype_classes") != {"param": "f16", "grad_reduce": "f32"}:
            raise ValueError(
                f"{pe_name} must explicitly freeze bfloat16 params and float32 reductions"
            )
        if pe.get("optimizer") != EXPECTED_OPTIMIZER:
            raise ValueError(f"{pe_name} must freeze the reviewed default AdamW state")
        identities = pe.get("identidades_arquitectonicas")
        expected_identities = {"D": "H*Dh"} if pe_name == "PE_dense" else {}
        if identities != expected_identities:
            raise ValueError(
                f"{pe_name} architectural identities: expected {expected_identities}, got {identities}"
            )
        symbols = pe.get("simbolos", {})
        if any(not isinstance(value, int) or value <= 0 for value in symbols.values()):
            raise ValueError(f"{pe_name} symbols must be positive integers")
        if symbols.get("L") != architecture["layers"]:
            raise ValueError(f"{pe_name} symbol L does not match architecture layers")
        if pe_name == "PE_dense" and symbols.get("D") != symbols.get("H") * symbols.get("Dh"):
            raise ValueError("PE_dense does not validate D=H*Dh")
        if pe_name == "PE_dense" and (
            architecture.get("fuse_qkv") is not True or symbols.get("Hkv") != symbols.get("H")
        ):
            raise ValueError("PE_dense must freeze fused QKV with Hkv=H")
        if pe_name == "PE_moe":
            mla = pe.get("dimensiones_mla", {})
            if set(mla) != {
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "v_head_dim",
                "kv_lora_rank",
            }:
                raise ValueError("PE_moe must freeze all reviewed MLA dimensions")
            expected_mla_symbols = {
                "Qn": mla["qk_nope_head_dim"],
                "Qr": mla["qk_rope_head_dim"],
                "Dv": mla["v_head_dim"],
                "Rkv": mla["kv_lora_rank"],
            }
            if {name: symbols.get(name) for name in expected_mla_symbols} != expected_mla_symbols:
                raise ValueError(
                    f"PE_moe MLA symbols do not match reviewed dimensions: {expected_mla_symbols}"
                )
            if (
                symbols.get("F") != architecture.get("moe_ffn_hidden")
                or symbols.get("Fd") != architecture.get("dense_ffn_hidden")
            ):
                raise ValueError("PE_moe F/Fd symbols must match MoE/dense FFN widths")
        if architecture["layers"] != 6:
            raise ValueError(f"{pe_name} candidate must use the six-layer debugmodel")
        if architecture["dense_layers"] + architecture["moe_layers"] != 6:
            raise ValueError(f"{pe_name} dense/MoE layer counts do not cover the model")
        if architecture["mtp_layers"] != 0:
            raise ValueError(f"{pe_name} candidate unexpectedly enables MTP")
        world_size = (
            overrides["data_parallel_replicate_degree"]
            * overrides["data_parallel_shard_degree"]
            * overrides["context_parallel_degree"]
            * overrides["tensor_parallel_degree"]
            * overrides["pipeline_parallel_degree"]
        )
        if world_size != expected_world_size:
            raise ValueError(f"{pe_name} world size: expected {expected_world_size}, got {world_size}")
        degrees = pe["grados"]
        expected_efsdp = (
            overrides["data_parallel_shard_degree"]
            * overrides["context_parallel_degree"]
            * overrides["tensor_parallel_degree"]
            // overrides["expert_parallel_degree"]
        )
        if pe_name == "PE_moe" and degrees["efsdp"] != expected_efsdp:
            raise ValueError(
                f"{pe_name} efsdp: expected {expected_efsdp}, got {degrees['efsdp']}"
            )
        degree_mapping = {
            "dp_r": overrides["data_parallel_replicate_degree"],
            "dp_s": overrides["data_parallel_shard_degree"],
            "cp": overrides["context_parallel_degree"],
            "tp": overrides["tensor_parallel_degree"],
            "ep": overrides["expert_parallel_degree"],
            "pp": overrides["pipeline_parallel_degree"],
        }
        for axis, configured in degree_mapping.items():
            declared = degrees[axis]
            normalized = None if configured == 1 else configured
            if declared != normalized:
                raise ValueError(
                    f"{pe_name} degree mismatch for {axis}: {declared} != {normalized}"
                )
        parts = overrides["module_fqns_per_model_part"]
        validate_partition(parts)
        if symbols.get("P") != len(parts):
            raise ValueError(f"{pe_name} P must equal the virtual pipeline stage count")
        schedule = overrides["pipeline_parallel_schedule"]
        expected_parts = overrides["pipeline_parallel_degree"] * (2 if schedule == "Interleaved1F1B" else 1)
        if len(parts) != expected_parts:
            raise ValueError(f"{pe_name} partition is incompatible with {schedule}")
        results[pe_name] = {
            "function_exists": True,
            "world_size": world_size,
            "pipeline_parts": len(parts),
            "partition_complete": True,
            "degrees_consistent": True,
            "architecture_consistent": True,
            "optimizer_consistent": True,
            "dispatcher_consistent": pe_name != "PE_moe" or overrides["moe_comm_backend"] == "standard",
        }
    return {
        "status": "VALID_CANDIDATE_NOT_FROZEN",
        "reference_sha": actual,
        "manifest_sha256": hashlib.sha256(canonical_bytes(manifest)).hexdigest(),
        "pes": results,
        "e0_closed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("e0-manifest-candidate.json"))
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        report = validate(args.reference_repo.resolve(), manifest)
    except (OSError, subprocess.CalledProcessError, SyntaxError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
