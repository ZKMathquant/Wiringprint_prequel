"""Versions of shared code exactly as used by notebook(s) i1, i2.

Kept apart from the canonical modules because the newer notebooks changed behaviour (see
docs/REFACTOR.md). Import from here ONLY in the notebooks named above; every other notebook
uses the canonical module of the same function name.

Notebook variables read live (set them in the notebook, after `wp.bind(globals())`):
    DEVICE
"""

import torch
import torch.nn.functional as F
from ..runtime import ambient as _nb, LATE as _LATE
from ..arch import get_layers, get_mlp
from ..condition import ConditionModel
from ..device import _amp_ctx, free_memory
from ..perturb import cpu_orthogonal
from ..stats import linear_cka


def rotation_angle_sweep_mlp(model, layer_idx, angles, seed=0, device='cpu'):
    '''Generator version -- yields (theta, ConditionModel) ONE AT A TIME so the
    caller can evaluate and drop each variant before the next is built, the
    same "one condition at a time" discipline Section 3 already uses.

    This is also what fixes the OOM the previous run hit here: the old version
    (a) built A, Q_full, and I as full fp32 d_ff x d_ff buffers directly ON
    THE GPU (cuSOLVER's QR only supports fp32/fp64, so that was already the
    largest per-call allocation in the whole notebook), and (b) returned a
    *list* of all len(angles) ConditionModels up front, so all of their
    variant tensors -- plus every theta's transient fp32 blend/Qb buffer --
    stayed reachable simultaneously. With three ~1.2-2.6B checkpoints already
    resident (~11.7GB/14.6GB used), that combination was enough to OOM on
    Gemma-2-2b's larger d_ff=9216. Here, Q_full/I are built once on CPU
    (cheap, and cuSOLVER's fp32-only restriction is moot there), and only the
    single current theta's small model-dtype Qb and its resulting per-layer
    variant tensors ever reach the GPU -- freed as soon as the caller moves on
    to the next theta.
    '''
    mlp_ref = get_mlp(model, layer_idx)
    d_ff = mlp_ref.gate_proj.out_features
    model_dtype = mlp_ref.gate_proj.weight.dtype

    Q_full = cpu_orthogonal(d_ff, seed=seed)
    I = torch.eye(d_ff, dtype=torch.float32)

    for theta in angles:
        # interpolate rotation generator in the Lie algebra sense: exponential map of
        # theta * logm(Q_full) would be ideal, but for a lightweight sweep we linearly
        # blend Q_full and I then re-orthogonalize -- a simple, inspectable proxy for
        # "how much rotation." Blend + re-QR both run on CPU; only the resulting
        # matrix, already cast to model_dtype (half the fp32 footprint), is moved
        # to `device` -- and only for the moment it's needed below.
        blend = (1 - theta) * I + theta * Q_full
        Qb_cpu, _ = torch.linalg.qr(blend)
        del blend
        Qb = Qb_cpu.to(device=device, dtype=model_dtype)
        del Qb_cpu
        mlp = get_mlp(model, layer_idx)
        param_refs = [(mlp.gate_proj, 'weight'), (mlp.up_proj, 'weight'), (mlp.down_proj, 'weight')]
        variants = [
            Qb.T @ mlp.gate_proj.weight.data,
            Qb.T @ mlp.up_proj.weight.data,
            mlp.down_proj.weight.data @ Qb,
        ]
        del Qb
        yield theta, ConditionModel(model, param_refs, variants, label=f'rotation_theta{theta}_L{layer_idx}')


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
