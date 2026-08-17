# Reproducibility and sealing

The preregistration is not active until E0 is closed. Before that point, the
document is a draft and no study outputs should be treated as evidence.

All generated study data must live outside the Git checkout. The `--out-root`
argument is mandatory and rejects paths inside the checkout; this is stronger
than relying on `.gitignore`.

The local structural calibration harness can be run without PyTorch:

```text
python scripts/e0_calibration.py --reference-repo <torchtitan-checkout>
```

Its output is explicitly `SYNTHETIC_STRUCTURAL_ONLY`; it does not close E0,
derive fingerprints from the PR population, or replace the real multi-GPU
validation.

A second calibration script checks real group formation (§8.3, point 2)
without a GPU, using torch's CPU `gloo` backend:

```text
python scripts/e0_mesh_validation.py \
    --pe-name PE_moe --dp-replicate 1 --dp-shard 2 --cp 1 --tp 2 --pp 2 --ep 2 \
    --world-size 8 --torchtitan-repo <torchtitan-checkout> --out-root <external-output>

python scripts/e0_mesh_validation.py \
    --pe-name PE_dense --dp-replicate 2 --dp-shard 2 --cp 2 --tp 2 --pp 2 --ep 1 \
    --world-size 32 --torchtitan-repo <torchtitan-checkout> --out-root <external-output>
```

It requires an external Python environment with CPU-only `torch` and
`spmd_types==0.2.3` installed (not vendored by this repository). Its output
is explicitly labelled `gloo (CPU) -- NOT NCCL/GPU`: it confirms the
reference code forms the declared communication groups and that real
collectives run over them, but it does not validate performance, bandwidth,
or behavior over real interconnects. It does not close E0 either.

An experimental, provenance-first inventory for §8.3 point 3 is available:

```bash
python scripts/e0_reference_extractor.py \
  --reference-repo /path/to/torchtitan-at-reference-head \
  --manifest e0-manifest-candidate.json \
  --output /external/path/e0-reference-inventory.json
```

This prototype statically inventories declarative `ShardingConfig` boundaries
and explicit communication calls for the dense and MoE reference paths. Its
output is a review queue, not a complete fingerprint: it deliberately reports
`e0_closed=false`, `e6_computed=false`, and makes no coverage claim until it is
cross-checked manually and against runtime traces. It never reads the PR
population, and rejects output paths inside the Planlock checkout.
The current Llama3/DeepSeek V3 routes are explicitly hypotheses: §1.0 does
not yet freeze each PE's `function_config`, `overrides`, or `hash_manifiesto`,
so the prototype reports that omission as a blocking gap rather than claiming
complete coverage or calculating E6.

The concrete proposal is stored in `e0-manifest-candidate.json` and can be
validated without importing TorchTitan:

```text
python scripts/e0_manifest.py --reference-repo <torchtitan-checkout>
```

It uses the six-layer `llama3_debugmodel` and `deepseek_v3_debugmodel`, fixes
every parallel degree and pipeline module partition explicitly, checks the
registry functions against the pinned HEAD, and emits a canonical SHA-256.
Its status remains `CANDIDATE_NOT_FROZEN` until the complete fingerprints and
runtime cross-check demonstrate that both proposed PEs are valid.
The extractor validates that manifest first, records its canonical hash, then
builds a static call graph from each model's sharding entrypoint. Candidates
are labelled `ACTIVE_STATIC`, `ACTIVE_MANIFEST`, `CONDITIONAL_RUNTIME`,
`UNREACHABLE_STATIC`, or `UNREACHABLE_MANIFEST`; unreachable candidates remain
in the audit output but cannot contribute to a future E6 calculation.
The prototype transition inventory counts each active `ShardingConfig`
boundary declaration once and collapses alternate backend implementations of
the same explicit communication helper into one logical template. These are
still candidate transitions, not the complete seven-field fingerprints.
Framework-generated communication is emitted separately with symbolic
multiplicity. Pipeline uses `P - 1` exactly as required by §1.7/§1.8 Q2.
FSDP/HSDP events retain their mesh group and payload class but remain marked
`REQUIRES_SEMANTIC_DECOMPOSITION` or `REQUIRES_RUNTIME_CROSSCHECK`; they are
not added to E6 until physical units are decomposed into the seven semantic
template fields.

HSDP mechanics can be cross-checked with a real four-process CPU/Gloo trace:

```text
python scripts/e0_hsdp_trace.py --output <new-external-directory>
python scripts/e0_reference_extractor.py \
  --reference-repo <torchtitan-checkout> \
  --manifest e0-manifest-candidate.json \
  --hsdp-trace <external-directory>/report.json \
  --output <external-inventory.json>
```

The trace runs FSDP2 on a `dp_replicate=2 × fsdp=2` mesh and requires profiler
evidence for all-gather, reduce-scatter, and all-reduce before upgrading the
HSDP candidates to `CONFIRMED_CPU_GLOO_MECHANICS`. This confirms the mechanism,
not the complete reference-model fingerprint or GPU/NCCL behavior.

The extractor also emits `dense_storage_semantics`, a logical-parameter
catalog for the Llama3 debugmodel. It separates parameters before FSDP
flattening, assigns their §1.3 roles, §1.6.4.A forms, TP storage placements,
and symbolic multiplicities. Parameter and reduction dtype classes are frozen
explicitly in the candidate manifest. During this decomposition E0 added
`2·L` to §1.7 because SwiGLU `w1` and `w3` collapse to the same structural
template in every layer.
`moe_storage_semantics` applies the same logical decomposition to the router,
shared experts, and routed grouped-GEMM parameters. Dense and sparse mesh
families remain distinct, and identical `w1/w3` expert templates use the
calibrated multiplicity `2·L_moe`.
Both storage catalogs now emit and validate the complete §1.6 tensor signature
for each logical parameter family: canonical semantic axes and expressions,
the manifest-frozen dtype class, and `tensor_class=param`. The validator rejects
unknown or repeated known axes, empty expressions, and unsupported dtype or
tensor classes. These are tensor signatures, not yet seven-field templates:
producer/consumer placements and framework transitions still require semantic
composition.
The extractor additionally emits `gradient_tensor_signatures`. Each logical
parameter family has one corresponding gradient family with the same semantic
form, role, TP component, and symbolic multiplicity; its dtype is the
manifest-frozen `grad_reduce` class and its tensor class is `grad`. This is a
provenance-preserving signature derivation only: it does not assume the
producer/consumer placement of an FSDP/HSDP reduction.
The candidate manifest also freezes the selected optimizer as fused AdamW with
AMSGrad disabled. `scripts/e0_optimizer_state_probe.py` materializes one real
optimizer step for both fp16 and bf16 parameters and requires `exp_avg` and
`exp_avg_sq` to preserve the parameter dtype and shape while `step` is a scalar
fp32 tensor. The extractor consequently emits 42
`optimizer_state_tensor_signatures`: two shaped moments plus one scalar step
for each of the 14 parameter families. These remain signature records, not
claims about FSDP/HSDP transition placements.

```text
python scripts/e0_optimizer_state_probe.py --output-root <external-directory>
```
The MoE candidate additionally freezes `moe_comm_backend=standard`, selecting
`AllToAllTokenDispatcher` at the pinned HEAD. The extractor emits five
`control_metadata_tensor_signatures` per logical MoE layer: top-k expert IDs,
the boolean routing map, expert-sorted token mapping, local and exchanged
expert counts, and the post-exchange permutation. IDs/counts are `i64`; the
routing map is `bool`. The pre-flatten top-k IDs preserve `[B,S,K]`, while
post-flatten token-slot mappings use calibrated `routed_item=B*S*K`.
`moe_routing_activation_tensor_signatures` separately records full and top-k
router scores (`f32`), reordered differentiable scores (`f32`), routed expert
inputs/outputs (manifest low precision), and the combined `[B,S,D]` output.
`dense_nonattention_activation_tensor_signatures` records the Llama embedding,
per-layer norms, attention boundary output, SwiGLU projections/product, rowwise
FFN output, final norm, and LMHead logits. It preserves the calibrated `2*L`
coefficient for the structurally identical `w1/w3` outputs. QKV projections,
head splits, positional tensors, and attention score/value forms are excluded
until their attention-specific identities are decomposed.
The dense manifest now also freezes `attn_backend=flex` and fused QKV. The
`dense_attention_activation_tensor_signatures` catalog records the fused
linear output with the phase-2 `output_feature=(H+2*Hkv)*Dh` fallback, then the
post-split query `[B,S,H,Dh]`, key/value `[B,S,Hkv,Dh]` (multiplicity `2*L`),
inner-attention output, and flattened residual form. It does not invent a
materialized attention-score tensor: FlexAttention may fuse that internal.
The first MLA audit intentionally emits `HUELLA_NO_DERIVABLE`. At the pinned
DeepSeek debugmodel, Q/K use a 192-wide head (`128+64`) while V uses 128, so the
single normative `Dh` symbol cannot represent both. The post-split KV latent
width 512 and the no-position/rotary components also lack normative symbols.
`mla_signature_audit` records these as E0-blocking failures and lists forbidden
shortcuts; no literal-by-value identity or completeness claim is permitted.

Typical workflow:

```text
python scripts/population.py --repo <torchtitan> --out-root <external-output>
python scripts/sample.py --out-root <external-output>
python scripts/anchor.py --out-root <external-output>
python scripts/make_pairs.py --repo <torchtitan> --out-root <external-output>
```

The `sealed/` directory contains the deblinding maps and must remain private
until the blind review is complete. Do not copy it into this repository or
publish it alongside the blinded pairs.

When E0 is closed:

1. record the final SHA-256 digest of `preregistro-huella-estructural-v14.md`;
2. record the final Git commit containing that exact document;
3. create an annotated, signed tag named `prereg-v14`;
4. publish the tag and digest together with the study record.

Until those steps happen, the repository must not claim that v14 is signed or
frozen.
