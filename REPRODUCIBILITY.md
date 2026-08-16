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
