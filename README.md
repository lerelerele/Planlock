# planlock

**A lockfile for the communication plan of a distributed training run. CI fails on regression, not on change.**

---

> **Status: not implemented.** `planlock` is gated on a preregistered falsification study that tests whether its core abstraction — a structural fingerprint of the communication plan — survives ordinary refactors of a real codebase. If the fingerprint is unstable, the project is abandoned rather than repaired. Nothing below has shipped. See [Status and validation](#status-and-validation) and the [preregistration](preregistro-huella-estructural-v14.md).

`scripts/` contains instrumentation for the falsification study, not the
`planlock` tool itself. Its sealed outputs must be generated outside this Git
checkout; see the `--out-root` option on the study scripts.

---

## What it does

`planlock` records the logical communication plan a distributed training run actually executes collectives, placements, and the transitions between them as a symbolic snapshot that survives refactors, renames, and changes of mesh degree.

Commit the snapshot. `planlock` fails CI when a change introduces a new collective, moves traffic across a slower interconnect, materializes a partial result earlier than a consumer requires, or adds expert dispatch. Improvements advance the snapshot on their own.

Every finding carries its epistemic status. An observation is never phrased as a guarantee.

## Quick start

```python
from planlock import CommPlanMode

with CommPlanMode(symbolic=True) as plan:
    loss = train_step(model, batch)
    loss.backward()

plan.assert_no_regression("tests/plans/moe_training.lock", ratchet=True)
```

From the command line:

```
planlock record train.py --entry train_step --out comm_plan.lock
planlock check  train.py --baseline comm_plan.lock
planlock diff   baseline.lock current.lock
planlock fix    train.py --baseline comm_plan.lock
```

## The ratchet

A baseline that fails on every difference gets `--update-baseline` bolted into the pipeline within a month, and then it protects nothing. `planlock` fails in one direction only.

| Diff | Result |
|---|---|
| No new collective sites, and traffic is equal or lower in every domain | **Passes.** The baseline advances automatically. |
| Any protected metric worsens | **Fails.** |
| Traffic drops but synchronization edges or numerics change | **Review.** Requires an explicit decision. |
| Not symbolically comparable | **Review.** |

Protected metrics are per-template and additive: symbolic payload per domain, logical transition count, collective type, own buffer volume, numerical equivalence class. Peak HBM is **not** among them it is global and temporal, evaluated once against the whole execution plan, because two locally dominant changes can compose into an out-of-memory run.

## Identity that survives refactors

The snapshot keys on structure, not on names. `model.layers.17.mlp.w2` is not an identity; renaming a module, changing depth, or moving from 8 to 16 GPUs must not invalidate the file.

A plan is a multiset of templates with symbolic multiplicity:

```
L × {
    RowLinear[B,S,D,F]
    Partial<Sum,tp>
    ReduceScatter<axis=S, group=tp>
    RMSNorm[B,S,D]
}
```

```
L_moe × {
    Route[T,K,E]
    Dispatch<ep>
    GroupedGEMM
    Combine<ep>
}
```

Mesh degrees and shapes stay symbolic, with guards:

```
D % TP == 0
E % EP == 0
S >= TP
TP within nvlink_domain
```

Dropped from the fingerprint: module names and paths, layer indices, source locations, rank IDs, operations that do not change placement, and alpha-renaming of dimensions and axes. One baseline therefore covers a family of configurations rather than a single one.

## Epistemic status on every finding

| Label | Meaning |
|---|---|
| `STRUCTURAL` | Holds for every execution of the captured graph that satisfies its guards |
| `BOUNDED` | Holds for any execution meeting a declared contract, such as `top_k` and `capacity_factor` |
| `OBSERVED` | A statistic gathered over N concrete batches |
| `ASSUMED` | Topology, length range, or policy supplied by the user |

```
[STRUCTURAL]
Every MoE block runs dispatch + combine over ep.

[BOUNDED]
With top_k=2 and capacity_factor=1.25:
tokens_per_expert <= the limit declared by the router.

[OBSERVED, n=256]
p99 imbalance      = 1.43
max expert load    = 1.81 x mean
token drop rate    = 0.07%

[ASSUMED]
sequence_length in [1024, 8192]
ep=16 crosses two NVLink domains over IB
```

## Sample finding

```
PL102  Partial materialized earlier than required
       RowLinear -> Norm, multiplicity L

       produced:  Partial<Sum,tp>
       requested: Replicate
       inserted:  AllReduce(tp)

       next compatible consumer accepts:
           Shard(sequence)

       alternative:
           Partial -> Shard(sequence)
           via ReduceScatter(tp)

       current cost:
           payload: B*S*D*sizeof(BF16)
           domain:  nvlink_0

       numerical class: ALGEBRAICALLY_EQUIVALENT_REASSOCIATED
       fix available
```

## Two speeds

The gate that runs on every pull request has to be cheap, or it stops being a gate.

| Per pull request — minutes, no cluster | Nightly or pre-release — cluster |
|---|---|
| Logical plan | Real batches |
| Symbolic shapes | Adversarial MoE routing scenarios |
| Structural peepholes | Rank-to-rank traffic matrices |
| Bounds derived from contracts | Measured bandwidth |
| Symbolic peak HBM | Compute/communication overlap |
| Diff against the lockfile | Routing, padding, token drop |
| | Physical memory |
| | The physical plan NCCL actually chose |

The cheap tier runs on fake tensors and single-process execution. It needs no `cluster.yaml`: it infers the device mesh where one exists, keeps degrees and payloads symbolic, marks the physical domain as `unknown`, and disables only the claims that depend on NVLink or InfiniBand. It still catches reshard ping-pong, premature materialization, and structural change.

## MoE

Expert parallelism is where the traffic is, and it is also where a single trace lies to you: dispatch volume depends on routing, and routing changes every batch.

| Execution | MoE coverage |
|---|---|
| Fake tensors, per PR | Structure and analytic bounds |
| Micro `ep=2`, per PR | The real dispatch/combine code path |
| Nightly, multi-GPU | Distribution, scale, performance |

The micro test uses deterministic routing over roughly 64 tokens: balanced, all-to-one-expert, one empty destination, capacity overflow, with and without token drop. Its results are labelled `MICRO_TRACE` and never presented as a guarantee about behaviour at scale.

The MoE snapshot records `top_k`, expert count and local experts, dropless versus capacity-limited, `capacity_factor` and drop policy, the rank-to-rank traffic matrix, tokens per expert, empty experts, imbalance statistics, padding introduced for grouped GEMM, local/intra-node/inter-node payload, dispatch and combine separately, and maximum reserved buffers which may reflect worst case rather than the observed batch.

## Rules

```
reshard-ping-pong                    Shard(A) -> Replicate -> Shard(A)
partial-materialized-too-early       Partial -> Replicate -> consumer accepting Shard
avoidable-all-gather                 AllGather -> operation that discards the gathered axis
large-unintended-replication         Replicate(large tensor) introduced at a boundary
collective-crosses-slow-domain       fast domain -> slow domain
forward-backward-placement-mismatch
per-layer-repeated-communication
new-expert-dispatch
```

## Non-goals

> `planlock` does not search for the globally optimal sharding plan. It verifies and compares the plan that ran, and applies a finite set of local transformations with provable preconditions.

It never says *"this collective is unnecessary."* It says *"this collective can be removed by this local equivalence, under these preconditions."*

Determining necessity in the general case is a sharding solver, and a sharding solver inside a linter is a five-year project that fails. Every rule above is a peephole: ping-pong is local, all-reduce to reduce-scatter looks one consumer ahead, oversized replication is a size check. None requires global search.

## Numerical equivalence

Every fix and every diff declares a class:

```
BITWISE_EQUIVALENT
ALGEBRAICALLY_EQUIVALENT_REASSOCIATED
TOLERANCE_VALIDATED
DISTRIBUTIONALLY_EQUIVALENT
UNKNOWN
```

For `all_reduce` versus `reduce_scatter + all_gather`, the honest output is: semantically equivalent, floating point reassociated, bitwise equality not guaranteed, validation required. Floating point addition is not associative, so changing the reduction tree can change bits while preserving the mathematics. The performance ratchet advances automatically only when the numerical class does not worsen.

## Environment drift

The plan can regress without a line of code changing. An NCCL upgrade, a driver bump, or a different environment variable moves traffic. `planlock` therefore splits the environment fingerprint and runs on a schedule, not only on pull requests.

```
execution_fingerprint     (causal — triggers revalidation)
    nccl version
    driver version
    cuda runtime
    pytorch / backend versions
    algorithm and protocol actually selected
    rank -> GPU -> NIC mapping
    NCCL options that affected selection

context_fingerprint       (contextual — recorded, never blocking)
    container digest
    operating system
    remaining environment variables
    runner identifier
```

```
code changes, environment stable    -> attribute to the PR
code stable, causal fingerprint moves -> revalidate; fail only if the new measurement breaks the contract
everything stable, performance moves  -> hardware, congestion, or noise
```

Declared topology is not trusted on its own. A probe runs the real collectives over the real groups and sizes the plan uses, and the report separates logical payload from estimated traffic a ring all-reduce does not move one copy of the tensor.

## Status and validation

Nothing here is built. The order of work is deliberately inverted from the usual one:

1. **Falsification first.** A preregistered blind study takes hard negatives from a real repository refactors that touch parallelization code, rename modules, or rewrite a block's forward path without changing the plan and tests whether the structural fingerprint stays identical across them. The specification is frozen and hashed before any pull request is opened. One false positive traced to intrinsic instability ends the project.
2. **Churn measurement.** If routine refactors move the fingerprint, `planlock` is worthless regardless of what else it catches. A tool whose baseline needs approving every week is a tool that gets switched off.
3. **Only then, code.**

Two limits are known in advance and are not negotiable:

- A manual study cannot establish the service level objective. Bounding the false-alarm rate at the level a merge gate requires needs several hundred consecutive clean qualifying pull requests. That measurement exists only in shadow mode, over a quarter or more, under a frozen version of the checker.
- **No gate is proposed before that shadow period completes.** The first deliverable is an opt-in test, not a blocking check.

The checked-in preregistration is currently a draft. There is no signed
`prereg-v14` tag yet; that tag and the final hash must be created only after
E0 closes. See [reproducibility](REPRODUCIBILITY.md) for the sealed-study
workflow.

## License

Code and study instrumentation are licensed under the [Apache License 2.0](LICENSE).
The preregistration is part of the repository record; its active version must
be treated as the one identified by the signed release tag once E0 is closed.
