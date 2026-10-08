"""Model-level evaluation: perplexity, next-token accuracy, behavioural batteries, calibration corpus, quantised checkpoints.

Notebook variables read live (set them in the notebook, after `wp.bind(globals())`):
    CALIBRATION_FALLBACK, COPY_STREAM, DEVICE, N_CALIBRATION_TARGET, QUANTIZED_CHECKPOINTS, QUANT_MODE, perturbation_records
"""

import numpy as np
import torch
import warnings
from .runtime import ambient as _nb, LATE as _LATE
from .device import _amp_ctx, to_device_async, tokenize_pinned
from .stats import expected_calibration_error


def next_token_accuracy(model, tokenizer, texts, device='cpu'):
    accs = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
            if ids.shape[1] < 2:
                continue
            logits = model(ids).logits
            preds = logits[0, :-1, :].argmax(dim=-1)
            targets = ids[0, 1:]
            accs.append((preds == targets).float().mean().item())
    return float(np.mean(accs))


def perplexity(model, tokenizer, texts, device='cpu'):
    losses = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
            if ids.shape[1] < 2:
                continue
            out = model(ids, labels=ids)
            losses.append(out.loss.item())
    return float(np.exp(np.mean(losses)))


def run_battery(model, tokenizer, texts, device='cpu', label='model', verbose=True):
    acc = next_token_accuracy(model, tokenizer, texts, device)
    ppl = perplexity(model, tokenizer, texts, device)
    ece = expected_calibration_error(model, tokenizer, texts, device)
    if verbose:
        print(f"{label:>28s} | accuracy={acc:.4f}  perplexity={ppl:9.3f}  ECE={ece:.4f}")
    return {'label': label, 'accuracy': acc, 'perplexity': ppl, 'ece': ece}


def logit_diff(logits, correct_id, wrong_id):
    return (logits[0, -1, correct_id] - logits[0, -1, wrong_id]).item()


def ood_perplexity_report(model, tokenizer, ood_corpus, device='cpu', label='model'):
    results = {}
    for domain, texts in ood_corpus.items():
        results[domain] = perplexity(model, tokenizer, texts, device)
    return results


def few_shot_accuracy(model, tokenizer, examples, device='cpu'):
    hits = 0
    with torch.no_grad():
        for prompt, target in examples:
            ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
            target_id = tokenizer.encode(target, add_special_tokens=False)[0]
            pred_id = model(ids).logits[0, -1, :].argmax(dim=-1).item()
            hits += int(pred_id == target_id)
    return hits / len(examples)


def get_perturbation_acc(name, label_substr):
    """Look up a condition's accuracy from `perturbation_records` (Part 2, Section 3) by a
    substring of its label, instead of re-deriving or hand-copying the number."""
    matches = [r['accuracy'] for r in _nb.perturbation_records[name] if label_substr in r['label']]
    if not matches:
        raise KeyError(f"no perturbation_records entry for {name} matching {label_substr!r}")
    return matches[0]


def simple_task_accuracy_batched(model, tokenizer, examples, device=_LATE):
    if device is _LATE: device = _nb.DEVICE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts, targets = zip(*examples)
    enc = tokenizer(list(prompts), return_tensors='pt', padding=True).to(device)
    target_ids = torch.tensor(
        [tokenizer(t, add_special_tokens=False).input_ids[0] for t in targets], device=device)
    with torch.inference_mode(), _amp_ctx():
        logits = model(enc.input_ids, attention_mask=enc.attention_mask).logits
    last_real = enc.attention_mask.sum(dim=1) - 1
    pred_ids = logits[torch.arange(logits.shape[0]), last_real].argmax(dim=-1)
    return (pred_ids == target_ids).float().mean().item()


def batched_behavioral_metrics(model, tokenizer, texts, batch_size=8, device=_LATE, use_compile=True):
    if device is _LATE: device = _nb.DEVICE
    model.eval()
    all_nll = []
    all_correct = []
    all_conf = []
    total_tokens = 0
    runner = model

    # Use compiled runner only when it is a distinct compiled object and the caller explicitly requests it.
    # Perturbation experiments may disable this if the compiled graph fails the mutation-consistency audit.
    if not use_compile:
        runner = model

    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start:start+batch_size])
        batch = tokenize_pinned(tokenizer, batch_texts, device=device)
        if device == "cuda":
            with torch.cuda.stream(_nb.COPY_STREAM):
                d_batch = to_device_async(batch, device)
            torch.cuda.current_stream().wait_stream(_nb.COPY_STREAM)
        else:
            d_batch = to_device_async(batch, device)

        with torch.inference_mode():
            out = runner(**d_batch)
            logits = out.logits[:, :-1, :]
            targets = d_batch["input_ids"][:, 1:]
            mask = d_batch.get("attention_mask", torch.ones_like(d_batch["input_ids"]))[:, 1:].bool()

            logp = torch.log_softmax(logits.float(), dim=-1)
            tok_nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            pred = logits.argmax(dim=-1)
            probs = torch.softmax(logits.float(), dim=-1)
            conf = probs.max(dim=-1).values

            all_nll.append(tok_nll[mask].detach().cpu())
            all_correct.append((pred[mask] == targets[mask]).detach().cpu())
            all_conf.append(conf[mask].detach().cpu())
            total_tokens += int(mask.sum().item())

        del batch, d_batch, out, logits, targets, mask, logp, pred, probs, conf

    nll = torch.cat(all_nll).numpy()
    correct = torch.cat(all_correct).numpy().astype(float)
    conf = torch.cat(all_conf).numpy()

    # Fixed ECE bins; no hidden tuning.
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())

    return {
        "accuracy_token_weighted": float(correct.mean()),
        "perplexity_token_weighted": float(np.exp(nll.mean())),
        "ece": float(ece),
        "n_tokens": total_tokens,
    }


def max_logit_diff(model_a, model_b, tokenizer, texts):
    vals = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(_nb.DEVICE)
        with torch.inference_mode():
            a = model_a(ids).logits
            b = model_b(ids).logits
        vals.append((a.float() - b.float()).abs().max().item())
    return float(max(vals))


def load_external_calibration_corpus(n_target=_LATE, seed=0):
    """Real external benchmark first (LAMBADA test: independent, well-known, same
    next-token-prediction task this notebook already measures everywhere else); an
    explicitly-logged larger hand-authored fallback only if that can't be reached.
    Returns (texts, source_label, is_real_benchmark)."""
    if n_target is _LATE: n_target = _nb.N_CALIBRATION_TARGET
    try:
        from datasets import load_dataset
        ds = load_dataset("lambada", split="test")
        rng = np.random.default_rng(seed)
        n = min(n_target, len(ds))
        idx = rng.choice(len(ds), size=n, replace=False)
        texts = [ds[int(i)]["text"] for i in idx]
        return texts, f"lambada-test (real external benchmark, n={n})", True
    except Exception as exc:
        warnings.warn(
            f"Could not load the LAMBADA benchmark ({type(exc).__name__}: {exc}). Falling back "
            f"to the {len(_nb.CALIBRATION_FALLBACK)}-sentence hand-authored bank. This changes the "
            f"calibration claim from 'benchmark-scale' to 'expanded hand-authored set' -- the "
            f"results table below records which path actually ran, so this is never silent."
        )
        return _nb.CALIBRATION_FALLBACK, f"hand-authored fallback (n={len(_nb.CALIBRATION_FALLBACK)})", False


def per_sentence_next_token_accuracy(model, tok, texts, batch_limit=64):
    vals = []
    for text in texts[:batch_limit] if batch_limit else texts:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=256).input_ids.to(_nb.DEVICE)
        if ids.shape[1] < 2:
            continue
        with torch.inference_mode():
            logits = model(ids).logits[0, :-1]
        pred = logits.argmax(-1)
        vals.append(float((pred == ids[0, 1:]).float().mean().item()))
    return np.array(vals)


def optional_gptq_validation(name):
    ckpt = _nb.QUANTIZED_CHECKPOINTS.get(name)
    if _nb.QUANT_MODE == "none" or not ckpt:
        print(f"{name}: GPTQ/AWQ validation skipped (QUANT_MODE={_nb.QUANT_MODE}, checkpoint={ckpt}).")
        return None
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
    except Exception as exc:
        raise RuntimeError("GPTQ validation requested but the installed Transformers stack lacks GPTQConfig.") from exc
    raise NotImplementedError(
        "Supply a verified AWQ/GPTQ checkpoint and its exact quantization metadata here. "
        "The notebook intentionally refuses to fabricate a quantization recipe because different GPTQ/AWQ "
        "checkpoints use different group sizes, zero-point conventions and kernels."
    )
