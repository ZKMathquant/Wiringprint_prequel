"""Multi-seed / multi-angle sweeps built from the pieces above.

Notebook variables read live (set them in the notebook, after `wp.bind(globals())`):
    CORPUS_HASH, DEVICE, DTYPE, EVAL_CORPUS, MANIFEST_VERSION, ROTATION_ANGLES, SEEDS
"""

import numpy as np
from .runtime import ambient as _nb, LATE as _LATE
from .device import free_memory
from .evals import batched_behavioral_metrics, next_token_accuracy
from .infra import cache_get, cache_key, cache_put
from .perturb import (
    make_aligned_mlp_permutation,
    make_unaligned_mlp_permutation,
    rotation_angle_sweep_mlp,
)


def mlp_seed_sweep(model, tokenizer, texts, layer_idx, n_seeds, device='cpu'):
    aligned_diffs, unaligned_diffs = [], []
    baseline_acc = next_token_accuracy(model, tokenizer, texts, device)
    for seed in range(n_seeds):
        a, _ = make_aligned_mlp_permutation(model, layer_idx, seed=seed, device=device)
        u, _ = make_unaligned_mlp_permutation(model, layer_idx, seed=seed, device=device)
        aligned_diffs.append(next_token_accuracy(a, tokenizer, texts, device) - baseline_acc)
        unaligned_diffs.append(next_token_accuracy(u, tokenizer, texts, device) - baseline_acc)
    return np.array(aligned_diffs), np.array(unaligned_diffs)


def rotation_seed_sweep(model, tokenizer, layer_idx, friendly_name, angles=_LATE, seeds=_LATE):
    if angles is _LATE: angles = _nb.ROTATION_ANGLES
    if seeds is _LATE: seeds = _nb.SEEDS
    rows = []
    for seed in seeds:
        # SPEED FIX (confirmed by two real Kaggle runs: the first took 4.8 hours before crashing
        # elsewhere, the second was cancelled by the user at 4.6 hours for taking too long). This
        # used to open a FRESH rotation_angle_sweep_mlp generator per (seed, theta) pair -- which
        # means cpu_orthogonal's d_ff x d_ff QR (deterministic given the seed, so genuinely
        # redundant across thetas within the same seed) was recomputed 6 times per seed instead
        # of once. Building one generator per seed with the FULL nonzero-theta list, and pulling
        # every theta from it, reuses that one construction across all 6 thetas -- a 6x cut in
        # calls to the single most expensive operation in this notebook, on top of the separate
        # GPU-QR speedup in rotation_angle_sweep_mlp itself. Still checks the cache per (seed,
        # theta) first, same as before, so a partially-cached run doesn't rebuild anything it
        # doesn't need to.
        nonzero_angles = [t for t in angles if t != 0.0]
        need_gen = any(
            cache_get(cache_key(manifest=_nb.MANIFEST_VERSION, model=model.config._name_or_path,
                                 layer=layer_idx, seed=seed, theta=t, corpus=_nb.CORPUS_HASH,
                                 metric="rotation_accuracy", dtype=str(_nb.DTYPE))) is None
            for t in nonzero_angles
        )
        gen = (rotation_angle_sweep_mlp(model, layer_idx, nonzero_angles, seed=seed, device=_nb.DEVICE)
               if need_gen else None)

        for theta in angles:
            # Cache key uses the HF repo path (model.config._name_or_path) -- stable identity for
            # caching purposes, independent of any display-name changes.
            key = cache_key(manifest=_nb.MANIFEST_VERSION, model=model.config._name_or_path,
                            layer=layer_idx, seed=seed, theta=theta, corpus=_nb.CORPUS_HASH,
                            metric="rotation_accuracy", dtype=str(_nb.DTYPE))
            cached = cache_get(key)
            if cached is not None:
                rows.append(cached)
                continue

            # Use the existing real-checkpoint rotation implementation. Each condition is built/evaluated/dropped
            # one at a time, so no sweep holds a family of variants in VRAM.
            if theta == 0.0:
                acc = batched_behavioral_metrics(model, tokenizer, _nb.EVAL_CORPUS, use_compile=False)["accuracy_token_weighted"]
            else:
                # Pulled from the one shared per-seed generator built above, not a fresh one.
                _, cond = next(gen)
                acc = batched_behavioral_metrics(cond, tokenizer, _nb.EVAL_CORPUS, use_compile=False)["accuracy_token_weighted"]
                del cond
            # BUG FIX (confirmed by a real Kaggle run): the "model" field previously stored
            # model.config._name_or_path (e.g. "Qwen/Qwen2.5-1.5B", the HF repo path), but every
            # downstream consumer of rotation_df (Sec 8.5's ANOVA, Sec 8.7's M7 sweep) filters by
            # the friendly MODEL_CONFIGS key (e.g. "Qwen2.5-1.5B"). The two never matched, so Sec
            # 8.5's per-model groups were silently EMPTY -- f_oneway/ttest_ind on empty arrays
            # produced exactly the observed "SmallSampleWarning" + F=nan, p=nan, with no exception
            # raised to flag it. (Sec 8.7's M7 cell had a defensive fallback for this exact
            # mismatch and so happened to still work -- a hint the bug was already half-visible
            # without anyone tracing it to its source.) Storing the friendly name directly here
            # fixes it at the source; the fallback in the M7 cell is now dead code but harmless
            # and left in place as a defensive no-op.
            rec = {"model": friendly_name, "layer": layer_idx,
                   "seed": seed, "theta": theta, "accuracy": float(acc)}
            cache_put(key, rec)
            rows.append(rec)
        # Release this seed's generator (and the Q_full/I buffers held in its frame, GPU or
        # CPU) before moving to the next seed, rather than letting up to 15 of them accumulate.
        del gen
        free_memory()
    return rows
