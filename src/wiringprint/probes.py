"""Hidden-state capture and mechanistic probes (CKA under rotation, dilution, causal tracing, redundancy).

Notebook variables read live (set them in the notebook, after `wp.bind(globals())`):
    DEVICE
"""

import numpy as np
import torch
import torch.nn.functional as F
from .runtime import ambient as _nb, LATE as _LATE
from .arch import eager_attention, get_layers, get_mlp
from .device import _amp_ctx, free_memory
from .evals import logit_diff
from .perturb import rotation_angle_sweep_mlp
from .stats import linear_cka


def collect_hidden_states(model, tokenizer, texts, layer, device='cpu'):
    rows = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
            out = model(ids, output_hidden_states=True)
            rows.append(out.hidden_states[layer][0].float().cpu().numpy())
    return np.concatenate(rows, axis=0)


def run_causal_tracing(model, tokenizer, pairs, device='cpu'):
    n_layers = len(get_layers(model))
    restoration_by_layer = np.zeros(n_layers)
    used_pairs = 0

    for clean_prompt, correct_ans, corrupted_prompt, wrong_ans in pairs:
        correct_id = tokenizer.encode(correct_ans, add_special_tokens=False)[0]
        wrong_id = tokenizer.encode(wrong_ans, add_special_tokens=False)[0]

        clean_ids = tokenizer(clean_prompt, return_tensors='pt').input_ids.to(device)
        corrupt_ids = tokenizer(corrupted_prompt, return_tensors='pt').input_ids.to(device)
        if clean_ids.shape[1] != corrupt_ids.shape[1]:
            continue
        used_pairs += 1

        with torch.no_grad():
            clean_out = model(clean_ids, output_hidden_states=True)
            clean_cache = clean_out.hidden_states
            clean_logits = clean_out.logits
            corrupt_logits = model(corrupt_ids).logits

        clean_diff = logit_diff(clean_logits, correct_id, wrong_id)
        corrupt_diff = logit_diff(corrupt_logits, correct_id, wrong_id)
        subject_pos = clean_ids.shape[1] - 2

        for layer in range(n_layers):
            def hook(module, inp, out, layer=layer):
                if isinstance(out, tuple):
                    out[0][:, subject_pos, :] = clean_cache[layer + 1][:, subject_pos, :]
                    return out
                out[:, subject_pos, :] = clean_cache[layer + 1][:, subject_pos, :]
                return out

            h = get_layers(model)[layer].register_forward_hook(hook)
            with torch.no_grad():
                patched_logits = model(corrupt_ids).logits
            h.remove()

            patched_diff = logit_diff(patched_logits, correct_id, wrong_id)
            denom = (clean_diff - corrupt_diff) if (clean_diff - corrupt_diff) != 0 else 1e-8
            restoration_by_layer[layer] += (patched_diff - corrupt_diff) / denom

    if used_pairs > 0:
        restoration_by_layer /= used_pairs
    return restoration_by_layer, used_pairs


def collect_pooled_hidden_states(model, tokenizer, texts, layer, device='cpu'):
    """Mean-pool each text's token dimension down to a single row before returning.

    BUG FIX (was: reusing collect_hidden_states here, which concatenates every
    token's hidden state into one big (total_tokens, hidden) matrix). That is
    correct for Step 5 / Section 5 above because those compare the SAME model
    against itself (baseline vs. a perturbed copy) under the SAME tokenizer, so
    the token axis is guaranteed to line up 1:1, token-for-token, across the two
    sides of the comparison.

    It is NOT valid here: Qwen/Gemma/Llama use three different tokenizers, so
    the same EVAL_CORPUS text produces a different number of tokens for each
    model (this is exactly what crashed with "size 91 is different from 85" --
    two families' concatenated token axes just don't have the same length, and
    even where lengths coincidentally matched, token i in one model's BPE
    vocabulary is not token i in another's, so no meaningful correspondence
    would exist to average over token-for-token anyway).

    The fix operates on a representation that IS well-defined across tokenizers:
    one pooled vector per example. Mean-pooling over the (model-specific-length)
    token axis collapses every text to a single hidden_dim vector, so the row
    axis becomes "which of the 8 EVAL_CORPUS texts" -- identical across all
    three families by construction -- while the column axis (hidden_dim) is
    allowed to differ freely between models, which linear_cka already supports.
    This is also strictly cheaper on GPU/host memory than the old per-token
    concatenation: output is (n_texts, hidden) instead of (total_tokens, hidden),
    and nothing extra is held on GPU -- pooling happens on the same per-text
    numpy array right after it's moved to CPU, with no extra forward passes.
    """
    rows = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
            out = model(ids, output_hidden_states=True)
            h = out.hidden_states[layer][0].float().cpu().numpy()  # (seq_len, hidden)
            rows.append(h.mean(axis=0))  # pool over tokens -> (hidden,)
    return np.stack(rows, axis=0)  # (n_texts, hidden) -- n_texts matches across families


def mlp_contribution_cos_batched(model, tokenizer, texts, layer_idx, device=_LATE):
    """Cosine similarity, at real (non-pad) token positions, between the decoder layer's
    ACTUAL output under the model's own unrotated MLP weights vs. a rotated variant, at each
    theta -- one batched forward pass per theta/condition on the real checkpoint."""
    if device is _LATE: device = _nb.DEVICE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(texts, return_tensors='pt', padding=True).to(device)
    mask = enc.attention_mask.bool()
    layer = get_layers(model)[layer_idx]

    def capture_layer_output(m):
        captured = {}
        def hook(mod, inp, out):
            captured['out'] = (out[0] if isinstance(out, tuple) else out).detach()
        h = layer.register_forward_hook(hook)
        with torch.inference_mode():
            m(enc.input_ids, attention_mask=enc.attention_mask)
        h.remove()
        return captured['out'][mask].float()

    base_out = capture_layer_output(model)
    results = {}
    for theta, cond in rotation_angle_sweep_mlp(model, layer_idx, [0.10, 0.25, 0.50, 0.75, 1.00],
                                                 seed=0, device=device):
        rot_out = capture_layer_output(cond)
        results[theta] = F.cosine_similarity(base_out, rot_out, dim=-1).mean().item()
        del rot_out
    return results


def dilution_probe_batched(model, tokenizer, texts, layer_idx, device=_LATE):
    """Real, per-theta measurement of (a) mean ||MLP branch output|| / ||residual stream
    entering that layer|| and (b) cosine similarity of the FULL decoder-layer output between
    the rotated and baseline conditions."""
    if device is _LATE: device = _nb.DEVICE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(texts, return_tensors='pt', padding=True).to(device)
    mask = enc.attention_mask.bool()
    layer = get_layers(model)[layer_idx]
    mlp = get_mlp(model, layer_idx)

    def run(m):
        captured = {}
        def hook_pre(mod, inp):
            captured['residual_in'] = inp[0].detach()
        def hook_out(mod, inp, out):
            captured['layer_out'] = (out[0] if isinstance(out, tuple) else out).detach()
        def hook_mlp(mod, inp, out):
            captured['mlp_out'] = out.detach()
        h1 = layer.register_forward_pre_hook(hook_pre)
        h2 = layer.register_forward_hook(hook_out)
        h3 = mlp.register_forward_hook(hook_mlp)
        with torch.inference_mode():
            m(enc.input_ids, attention_mask=enc.attention_mask)
        h1.remove(); h2.remove(); h3.remove()
        return captured

    base = run(model)
    r = base['residual_in'].float().norm(dim=-1)[mask]
    base_layer_out = base['layer_out'][mask].float()

    results = {}
    for theta, cond in rotation_angle_sweep_mlp(model, layer_idx, [0.10, 0.25, 0.50, 0.75, 1.00],
                                                 seed=0, device=device):
        rot = run(cond)
        m_norm = rot['mlp_out'].float().norm(dim=-1)[mask]
        ratio = float((m_norm / (r + 1e-8)).mean().item())
        cos = F.cosine_similarity(base_layer_out, rot['layer_out'][mask].float(), dim=-1).mean().item()
        results[theta] = dict(ratio=ratio, cos=cos)
        del rot
    return results


def capture_layer_output(m, layer_module, enc, mask_pad, context=None):
    captured = {}
    def hook(mod, inp, out):
        captured['out'] = (out[0] if isinstance(out, tuple) else out).detach()
    h = layer_module.register_forward_hook(hook)
    with torch.inference_mode():
        if context is None:
            m(enc.input_ids, attention_mask=enc.attention_mask)
        else:
            with context:
                m(enc.input_ids, attention_mask=enc.attention_mask)
    h.remove()
    return captured['out'][mask_pad].float()


def find_top_redundant_pairs(model, layer_idx, top_k=64, device='cpu'):
    """Cosine similarity between REAL down_proj neuron 'output directions' (columns of the
    actual trained down_proj weight matrix) -- the top_k most mutually-similar (i, j) pairs are
    read directly off the real weights, not constructed by hand."""
    mlp = get_mlp(model, layer_idx)
    W = mlp.down_proj.weight.data.float().to(device)      # (d_model, d_ff)
    Wn = W / (W.norm(dim=0, keepdim=True) + 1e-8)
    sim = (Wn.T @ Wn).cpu()
    sim.fill_diagonal_(-2.0)
    best_sim, best_idx = sim.max(dim=1)
    order = torch.argsort(best_sim, descending=True)
    seen, pairs = set(), []
    for idx in order.tolist():
        j = int(best_idx[idx].item())
        key = tuple(sorted((idx, j)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((key[0], key[1], float(sim[key[0], key[1]].item())))
        if len(pairs) >= top_k:
            break
    return pairs, sim


def down_proj_effective_rank(model, layer_idx):
    mlp = get_mlp(model, layer_idx)
    W = mlp.down_proj.weight.data.float().to('cpu')
    s = torch.linalg.svdvals(W)
    eff_rank = float((s.sum() ** 2 / (s ** 2).sum()).item())
    return s.numpy(), eff_rank


def capture_mlp_hidden(model, tokenizer, texts, layer_idx, device=_LATE):
    """Captures the REAL post-activation hidden state h = act_fn(gate_proj(x)) * up_proj(x) --
    down_proj's actual input -- via a forward pre-hook on the real down_proj, at real (non-pad)
    token positions, for one forward pass."""
    if device is _LATE: device = _nb.DEVICE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(texts, return_tensors='pt', padding=True).to(device)
    mask = enc.attention_mask.bool()
    mlp = get_mlp(model, layer_idx)
    captured = {}
    def hook(mod, inp):
        captured['h'] = inp[0].detach()
    handle = mlp.down_proj.register_forward_pre_hook(hook)
    with torch.inference_mode():
        model(enc.input_ids, attention_mask=enc.attention_mask)
    handle.remove()
    return captured['h'][mask].float()   # (n_real_tokens, d_ff)


@torch._dynamo.disable
def _run_with_attentions(model, input_ids, attention_mask):
    return model(input_ids, attention_mask=attention_mask, output_attentions=True)


def attention_entropy_batched(model, tokenizer, texts, layer_idx, device=_LATE):
    """Mean entropy (nats) of the attention distribution at `layer_idx`, over real (non-pad)
    query positions, heads, and texts -- one batched forward pass."""
    if device is _LATE: device = _nb.DEVICE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(list(texts), return_tensors='pt', padding=True).to(device)
    with torch.inference_mode(), _amp_ctx(), eager_attention(model):
        out = _run_with_attentions(model, enc.input_ids, enc.attention_mask)
    if out.attentions is None or out.attentions[layer_idx] is None:
        raise RuntimeError(
            f"output_attentions still returned None at layer {layer_idx} even under eager "
            f"attention -- check model.config._attn_implementation actually reads back as "
            f"'eager' (print it), and that this transformers version supports output_attentions "
            f"under eager for this architecture.")
    attn = out.attentions[layer_idx]
    p = attn.float().clamp_min(1e-12)
    ent = -(p * p.log()).sum(dim=-1)
    mask = enc.attention_mask.bool().unsqueeze(1).expand_as(ent)
    return float(ent[mask].mean().item())


def layerwise_cka_under_rotation_batched(model, tokenizer, texts, rotated_layer_idx, seed=0, device=_LATE):
    """For each theta in the Part-4 sweep, collects every layer's hidden states (ONE batched,
    padded forward pass per theta) and CKA's against the theta=0 baseline, layer by layer."""
    if device is _LATE: device = _nb.DEVICE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(list(texts), return_tensors='pt', padding=True).to(device)
    mask = enc.attention_mask.bool()

    def collect_all_layers(m):
        with torch.inference_mode(), _amp_ctx():
            out = m(enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True)
        acts = {li: hs[mask].float().cpu().numpy() for li, hs in enumerate(out.hidden_states)}
        del out
        return acts

    base_acts = collect_all_layers(model)
    free_memory()
    results = {}
    for theta, cond in rotation_angle_sweep_mlp(model, rotated_layer_idx,
                                                 [0.10, 0.25, 0.50, 0.75, 1.00], seed=seed, device=device):
        rot_acts = collect_all_layers(cond)
        results[theta] = {li: linear_cka(base_acts[li], rot_acts[li]) for li in base_acts}
        del rot_acts
        free_memory()
    return results
