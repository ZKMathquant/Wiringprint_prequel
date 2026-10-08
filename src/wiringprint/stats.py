"""Metrics that need no model: CKA, Procrustes, principal angles, bootstrap, calibration error, hashing.
"""

import hashlib
import json
import numpy as np
import torch
from scipy import stats as scipy_stats
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment


def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:24]


def expected_calibration_error(model, tokenizer, texts, device='cpu', n_bins=10):
    confidences, corrects = [], []
    model.eval()
    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
            if ids.shape[1] < 2:
                continue
            logits = model(ids).logits[0, :-1, :]
            targets = ids[0, 1:]
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            confidences.extend(conf.tolist())
            corrects.extend((pred == targets).tolist())
    confidences = np.array(confidences)
    corrects = np.array(corrects).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(confidences)) * abs(corrects[mask].mean() - confidences[mask].mean())
    return float(ece)


def linear_cka(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = ((X.T @ Y) ** 2).sum()
    den = np.sqrt(((X.T @ X) ** 2).sum() * ((Y.T @ Y) ** 2).sum())
    return float(num / den) if den > 0 else 0.0


def mean_ci(values, confidence=0.95):
    mean = values.mean()
    sem = scipy_stats.sem(values) if len(values) > 1 else 0.0
    ci = scipy_stats.t.interval(confidence, len(values) - 1, loc=mean, scale=sem) if sem > 0 else (mean, mean)
    return float(mean), (float(ci[0]), float(ci[1]))


def procrustes_distance(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    X = X / (np.linalg.norm(X) + 1e-8)
    Y = Y / (np.linalg.norm(Y) + 1e-8)
    R, _ = orthogonal_procrustes(X, Y)
    return float(np.linalg.norm(X @ R - Y))


def top_k_principal_angles(hidden_a, hidden_b, k=10):
    a = hidden_a - hidden_a.mean(axis=0, keepdims=True)
    b = hidden_b - hidden_b.mean(axis=0, keepdims=True)
    _, _, Va = np.linalg.svd(a, full_matrices=False)
    _, _, Vb = np.linalg.svd(b, full_matrices=False)
    k = min(k, Va.shape[0], Vb.shape[0])
    s = np.linalg.svd(Va[:k].T.T @ Vb[:k].T, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


def permutation_aligned_cos(h_base, h_treated, max_tokens=512):
    """Best-permutation-aligned per-neuron cosine similarity (Hungarian algorithm on the
    d_ff x d_ff neuron-vs-neuron similarity matrix, using each neuron's activation across
    tokens as its signature), subsampled to at most max_tokens rows for tractability."""
    n = min(h_base.shape[0], h_treated.shape[0], max_tokens)
    hb = h_base[:n].cpu().numpy(); hr = h_treated[:n].cpu().numpy()
    hb_n = hb / (np.linalg.norm(hb, axis=0, keepdims=True) + 1e-8)
    hr_n = hr / (np.linalg.norm(hr, axis=0, keepdims=True) + 1e-8)
    sim = hb_n.T @ hr_n
    row_ind, col_ind = linear_sum_assignment(-sim)
    return float(sim[row_ind, col_ind].mean()), float(np.diagonal(sim).mean())


# Helper for §8.7: identifies each model's own empirically-best-accuracy NONZERO rotation
# angle from a rotation sweep (excludes theta=0, which is the unperturbed baseline by
# construction and is not a "recovery" point). This was called but never defined in the
# supplied notebook -- added here to make the notebook runnable standalone.
def best_recovery_angle(mean_rows):
    nonzero = [r for r in mean_rows if r["theta"] > 0]
    if not nonzero:
        raise ValueError("No nonzero-theta rows in rotation sweep; cannot compute a recovery angle.")
    return max(nonzero, key=lambda r: r["accuracy"])["theta"]


def bootstrap_core_vs_external(core_acc, external_acc, n_boot=5000, seed=0):
    """The statistically correct form of 'does the 8-sentence corpus generalize': resample the
    LARGE external set at the SAME size as the small core corpus, many times, and ask whether the
    core corpus's actual mean accuracy falls inside the resulting distribution -- rather than
    forcing a fragile Pearson r out of only 8 paired points."""
    rng = np.random.default_rng(seed)
    n_core = len(core_acc)
    core_mean = float(core_acc.mean())
    boot_means = np.array([
        rng.choice(external_acc, size=n_core, replace=True).mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    within_ci = bool(lo <= core_mean <= hi)
    p_two_sided = float(np.mean(np.abs(boot_means - boot_means.mean()) >= abs(core_mean - boot_means.mean())))
    return {
        "core_mean": core_mean,
        "external_mean": float(external_acc.mean()),
        "n_core": int(n_core),
        "n_external": int(len(external_acc)),
        "boot_ci_2_5": float(lo),
        "boot_ci_97_5": float(hi),
        "core_within_external_boot_ci_95": within_ci,
        "empirical_two_sided_p": p_two_sided,
    }
