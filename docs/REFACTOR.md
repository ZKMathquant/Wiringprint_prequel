# Shared code: `src/wiringprint`

The five notebooks used to carry ~5,800 lines of helper functions, most of them copied between notebooks. Those helpers now live once in `src/wiringprint/`; each notebook imports what it uses. Notebooks keep their experiment cells, settings and any helper that is specific to them.

## How it runs

* **Kaggle:** `tools/runner.py` puts the *pinned* checkout's `src/` on `PYTHONPATH` and takes the notebook from the same commit, so notebook and shared code can never come from different commits. `run_manifest.json` records `src_sha256`.
* **Local / Colab:** the first injected cell of each notebook walks up from the working directory to find `src/`; or `pip install -e .`.
* **Before spending GPU hours:** `python tools/check_repo.py` (also runs on every push via the *Check repo* workflow) verifies every `from wiringprint... import name` resolves. The Kaggle *self-check* additionally imports every module on Kaggle's real torch/transformers stack.

## What was changed, precisely

A function was moved only if moving it provably changes nothing. Two mechanical edits are applied, and nothing else:

1. **Notebook variables become `_nb.NAME`.** A notebook helper that read a notebook variable (`DEVICE`, `DTYPE`, ...) as a global now reads it through `wiringprint.runtime` at call time from the namespace given to `wp.bind(globals())`. Same value, same moment as before.
2. **`device=DEVICE`-style defaults become `device=_LATE`** plus one line `if device is _LATE: device = _nb.DEVICE`. Python evaluates defaults when the `def` runs, which in a package is import time. This is applied only where the variable is assigned before the def and never again after it, so the value is identical at call time.

Everything else (bodies, comments, docstrings) is copied verbatim. A build-time verifier re-derived, from the files alone, that for all 310 moved definitions the module AST equals the notebook AST under exactly these two edits, that every other top-level statement of every notebook is unchanged and in the same order, that no new undefined names appear, and that all imports resolve.

## Modules

### `wiringprint.arch`
Architecture access: layers, MLP/attention blocks, head counts, parameter references, attention/RoPE switches.

| function | used by |
|---|---|
| `attn_param_refs` | i1, i2, i3, i4 |
| `count_rope_modules` | i2, i3, i4 |
| `disable_rope` | i2, i3, i4 |
| `eager_attention` | i1, i2, i3, i4 |
| `full_model_param_refs` | i1, i2, i3, i4 |
| `get_attn` | i1, i2, i3, i4 |
| `get_d_ff` | i1, i2, i3, i4 |
| `get_head_dim` | i1, i2, i3, i4 |
| `get_layers` | i1, i2, i3, i4 |
| `get_mlp` | i1, i2, i3, i4 |
| `get_num_heads` | i1, i2, i3, i4 |
| `get_num_kv_heads` | i1, i2, i3, i4 |
| `mlp_param_refs` | i1, i2, i3, i4 |
| `robust_head_config` | i1, i2, i3, i4 |

### `wiringprint.device`
Device, dtype and GPU-memory helpers (OOM guard, autocast context, pinned/async transfers).

| function | used by |
|---|---|
| `_amp_ctx` | i1, i2, i3, i4 |
| `cuda_oom_guard` | i1, i2, i3, i4 |
| `free_memory` | i1, i2, i3, i4 |
| `pick_dtype` | i1, i2 |
| `print_gpu_mem` | i1, i2, i3, i4 |
| `to_device_async` | i3, i4 |
| `tokenize_pinned` | i3, i4 |

### `wiringprint.stats`
Metrics that need no model: CKA, Procrustes, principal angles, bootstrap, calibration error, hashing.

| function | used by |
|---|---|
| `best_recovery_angle` | i3, i4 |
| `bootstrap_core_vs_external` | i3, i4 |
| `expected_calibration_error` | i1, i2, i3, i4 |
| `linear_cka` | i1, i2, i3, i4 |
| `mean_ci` | i0, i1, i2, i3, i4 |
| `permutation_aligned_cos` | i2, i3, i4 |
| `procrustes_distance` | i1, i2, i3, i4 |
| `stable_hash` | i3, i4 |
| `top_k_principal_angles` | i1, i2, i3, i4 |

### `wiringprint.condition`
ConditionModel: a cheap, reversible 'perturbed model' view that swaps parameter tensors in place.

| function | used by |
|---|---|
| `ConditionModel` | i1, i2, i3, i4 |
| `_smoke_test_condition_model` | i1, i2, i3, i4 |
| `bypass_module` | i1, i2, i3, i4 |
| `check_scaling_symmetry` | i1, i2, i3, i4 |

### `wiringprint.evals`
Model-level evaluation: perplexity, next-token accuracy, behavioural batteries, calibration corpus, quantised checkpoints.

| function | used by |
|---|---|
| `batched_behavioral_metrics` | i3, i4 |
| `few_shot_accuracy` | i1, i2, i3, i4 |
| `get_perturbation_acc` | i2, i3, i4 |
| `load_external_calibration_corpus` | i3, i4 |
| `logit_diff` | i0, i1, i2, i3, i4 |
| `max_logit_diff` | i3, i4 |
| `next_token_accuracy` | i0, i1, i2, i3, i4 |
| `ood_perplexity_report` | i1, i2, i3, i4 |
| `optional_gptq_validation` | i3, i4 |
| `per_sentence_next_token_accuracy` | i3, i4 |
| `perplexity` | i0, i1, i2, i3, i4 |
| `run_battery` | i1, i2, i3, i4 |
| `simple_task_accuracy_batched` | i1, i2, i3, i4 |

### `wiringprint.perturb`
Weight-space transformations (permutations, rotations, noise, knock-outs, fine-tuning) producing ConditionModels.

| function | used by |
|---|---|
| `build_block_permutation` | i1, i2, i3, i4 |
| `cpu_orthogonal` | i1, i2, i3, i4 |
| `finetune_in_place` | i1, i2, i3, i4 |
| `head_knockout` | i2, i3, i4 |
| `head_knockout_mask` | i2, i3, i4 |
| `make_aligned_mlp_permutation` | i1, i2, i3, i4 |
| `make_aligned_mlp_permutation_real` | i1, i2, i3, i4 |
| `make_gateonly_mlp_permutation` | i1, i2, i3, i4 |
| `make_gaussian_noise_mlp` | i1, i2, i3, i4 |
| `make_global_rotation` | i1, i2, i3, i4 |
| `make_local_orthogonal_mlp` | i1, i2, i3, i4 |
| `make_neuron_deletion_mlp` | i1, i2, i3, i4 |
| `make_neuron_knockout` | i2, i3, i4 |
| `make_rank_truncated_downproj` | i2, i3, i4 |
| `make_unaligned_mlp_permutation` | i1, i2, i3, i4 |
| `partial_violation_mlp_permutation` | i1, i2, i3, i4 |
| `permute_attention_heads_gqa` | i1, i2, i3, i4 |
| `rotation_angle_sweep_mlp` | i3, i4 |

### `wiringprint.probes`
Hidden-state capture and mechanistic probes (CKA under rotation, dilution, causal tracing, redundancy).

| function | used by |
|---|---|
| `_run_with_attentions` | i3, i4 |
| `attention_entropy_batched` | i3, i4 |
| `capture_layer_output` | i2, i3, i4 |
| `capture_mlp_hidden` | i2, i3, i4 |
| `collect_hidden_states` | i1, i2, i3, i4 |
| `collect_pooled_hidden_states` | i1, i2, i3, i4 |
| `dilution_probe_batched` | i3, i4 |
| `down_proj_effective_rank` | i2, i3, i4 |
| `find_top_redundant_pairs` | i2, i3, i4 |
| `layerwise_cka_under_rotation_batched` | i3, i4 |
| `mlp_contribution_cos_batched` | i3, i4 |
| `run_causal_tracing` | i1, i2, i3, i4 |

### `wiringprint.sweeps`
Multi-seed / multi-angle sweeps built from the pieces above.

| function | used by |
|---|---|
| `mlp_seed_sweep` | i1, i2, i3, i4 |
| `rotation_seed_sweep` | i3, i4 |

### `wiringprint.infra`
Performance infrastructure: torch.compile, attention backend choice, generation-cache benchmark, on-disk result cache.

| function | used by |
|---|---|
| `_has_pkg` | i3, i4 |
| `apply_attention_backend` | i3, i4 |
| `benchmark_generation_cache` | i3, i4 |
| `cache_get` | i3, i4 |
| `cache_key` | i3, i4 |
| `cache_put` | i3, i4 |
| `choose_attention_backend` | i3, i4 |
| `try_compile_model` | i3, i4 |

## Versions kept apart (important)

Some helpers **changed behaviour between notebook generations**. Merging them silently would change results of runs already archived, so older versions live in `wiringprint.variants.*` and are imported only by the notebooks that originally used them.

### `wiringprint.variants.nb_i1_i2`  (notebooks i1, i2)

| function | relation to the canonical version |
|---|---|
| `dilution_probe_batched` | identical code, but calls the older `rotation_angle_sweep_mlp` below, so its results differ |
| `layerwise_cka_under_rotation_batched` | identical code, but calls the older `rotation_angle_sweep_mlp` below, so its results differ |
| `mlp_contribution_cos_batched` | identical code, but calls the older `rotation_angle_sweep_mlp` below, so its results differ |
| `rotation_angle_sweep_mlp` | different code (see the two sources) |

The one substantive difference: `rotation_angle_sweep_mlp` in i1/i2 builds the blended rotation and its QR on **CPU**; i3/i4 do it on **GPU** (a speed fix for a 4.8 h step) with a CPU fallback. QR on different devices can differ in the last floating-point digits.

## Deliberately left in the notebooks

| notebook | defs kept | why |
|---|---|---|
| i0 | 64 | single-notebook version |
| i0 | 16 | cannot be moved safely: defined more than once in this notebook with different bodies |
| i0 | 5 | cannot be moved safely: callee 'build_head_block_permutation' is redefined differently within this notebook |
| i0 | 2 | cannot be moved safely: callee 'expected_calibration_error' is redefined differently within this notebook; defined more than once in this notebook with different bo |
| i0 | 2 | cannot be moved safely: callee 'linear_cka_stable' is redefined differently within this notebook |
| i1 | 11 | single-notebook version |
| i2 | 1 | single-notebook version |
| i3 | 1 | cannot be moved safely: scope-dependent builtin (globals) |
| i3 | 1 | cannot be moved safely: callee 'cache_capabilities' cannot be shared; scope-dependent builtin (__import__) |
| i3 | 1 | cannot be moved safely: scope-dependent builtin (__import__) |
| i4 | 1 | cannot be moved safely: scope-dependent builtin (globals) |
| i4 | 1 | cannot be moved safely: callee 'cache_capabilities' cannot be shared; scope-dependent builtin (__import__) |
| i4 | 1 | cannot be moved safely: scope-dependent builtin (__import__) |

`i0` is the original GPT-2 notebook: it redefines several helpers several times with different bodies and uses different internals, so only 14 of its 103 definitions are shareable without changing what it computes. `cache_capabilities`/`package_version` (i3, i4) inspect their own namespace (`globals()`, `__import__`) and would behave differently inside a package.

## Notebook variables the shared code reads

| variable | read by modules |
|---|---|
| `ATTN_IMPL` | infra |
| `CACHE_DIR` | infra |
| `CALIBRATION_FALLBACK` | evals |
| `COMPILE_ENABLED` | infra |
| `COMPILE_MODE` | infra |
| `COPY_STREAM` | evals |
| `CORPUS_HASH` | sweeps |
| `DEVICE` | device, evals, infra, probes, sweeps, variants.nb_i1_i2 |
| `DTYPE` | device, sweeps |
| `EVAL_CORPUS` | infra, sweeps |
| `GPU_HEADROOM_FRAC` | infra |
| `MANIFEST_VERSION` | sweeps |
| `N_CALIBRATION_TARGET` | evals |
| `PRECISION_MODE` | device |
| `QUANTIZED_CHECKPOINTS` | evals |
| `QUANT_MODE` | evals |
| `ROTATION_ANGLES` | sweeps |
| `SEEDS` | sweeps |
| `_EST_TOTAL_PARAMS` | device |
| `perturbation_records` | evals |

Call `wp.bind(globals())` before using shared code (the injected cell does). A missing variable raises `NameError`, as before.

## Adding or changing shared code
* Edit the function in `src/wiringprint/<module>.py`; every notebook that imports it changes with it. **Frozen runs under `runs/` are unaffected.**
* If a change should not affect an earlier notebook, copy the old version into `wiringprint/variants/` first and point that notebook at it.
* Prefer passing values as arguments over adding new `_nb.X` reads; `_nb` exists to keep the original helpers byte-identical, not as a design to extend.
