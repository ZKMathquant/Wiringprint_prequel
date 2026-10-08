"""Architecture access: layers, MLP/attention blocks, head counts, parameter references, attention/RoPE switches.
"""

import torch
from contextlib import contextmanager


def get_layers(model):
    return model.model.layers


def get_mlp(model, layer_idx):
    return model.model.layers[layer_idx].mlp


def get_attn(model, layer_idx):
    return model.model.layers[layer_idx].self_attn


def get_num_heads(model):
    return model.config.num_attention_heads


def get_num_kv_heads(model):
    # ROBUST HEAD-COUNT DETECTION (generalizes the original's GPT-NeoX/Pythia annex):
    # newer configs expose num_key_value_heads directly; fall back to num_attention_heads
    # (== MHA, group size 1) if the attribute is absent, as in some older configs.
    return getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)


def get_head_dim(model):
    cfg = model.config
    if hasattr(cfg, 'head_dim') and cfg.head_dim is not None:
        return cfg.head_dim
    return cfg.hidden_size // cfg.num_attention_heads


def get_d_ff(model, layer_idx):
    return get_mlp(model, layer_idx).gate_proj.out_features


def mlp_param_refs(mlp):
    """(module, attr) pairs for an MLP block's weight/bias tensors."""
    refs = [(mlp.gate_proj, 'weight'), (mlp.up_proj, 'weight'), (mlp.down_proj, 'weight')]
    if mlp.gate_proj.bias is not None:
        refs.append((mlp.gate_proj, 'bias'))
    if mlp.up_proj.bias is not None:
        refs.append((mlp.up_proj, 'bias'))
    if mlp.down_proj.bias is not None:
        refs.append((mlp.down_proj, 'bias'))
    return refs


def attn_param_refs(attn):
    """(module, attr) pairs for an attention block's Q/K/V/O weight/bias tensors."""
    refs = [(attn.q_proj, 'weight'), (attn.k_proj, 'weight'), (attn.v_proj, 'weight'), (attn.o_proj, 'weight')]
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
        if proj.bias is not None:
            refs.append((proj, 'bias'))
    return refs


def full_model_param_refs(model):
    """(module, attr) pairs spanning every layer's attention + MLP weights, plus
    embeddings and (if untied) lm_head -- used only by the global residual-stream
    rotation, which genuinely touches (almost) every tensor in the model. Still
    avoids duplicating the nn.Module graph itself, and (unlike the per-layer
    conditions) is meant to be used transiently, one at a time."""
    refs = [(model.model.embed_tokens, 'weight')]
    for layer in get_layers(model):
        refs.extend(attn_param_refs(layer.self_attn))
        refs.extend(mlp_param_refs(layer.mlp))
    if hasattr(model, 'lm_head') and not getattr(model.config, 'tie_word_embeddings', False):
        refs.append((model.lm_head, 'weight'))
    return refs


def robust_head_config(model):
    cfg = model.config
    n_heads = getattr(cfg, 'num_attention_heads', None)
    n_kv = getattr(cfg, 'num_key_value_heads', None)
    head_dim = getattr(cfg, 'head_dim', None)

    if n_heads is None:
        raise ValueError("Could not find num_attention_heads on this config.")
    if n_kv is None:
        n_kv = n_heads  # MHA fallback
    if head_dim is None:
        head_dim = cfg.hidden_size // n_heads

    assert n_heads % n_kv == 0, f"num_attention_heads ({n_heads}) not divisible by num_key_value_heads ({n_kv})"
    return dict(num_heads=n_heads, num_kv_heads=n_kv, head_dim=head_dim, group_size=n_heads // n_kv)


@contextmanager
def disable_rope(model):
    """Monkey-patches every real RotaryEmbedding submodule's forward to null out the
    position-dependent rotation it computes (cos->1, sin->0) for the duration of the context --
    a real causal ablation of the model's own rotary-embedding module. Restores every patched
    forward on exit, even on exception."""
    patched = []
    for _, module in model.named_modules():
        if 'RotaryEmbedding' in type(module).__name__:
            original_forward = module.forward
            def make_identity(orig):
                def _identity_forward(*args, **kwargs):
                    cos, sin = orig(*args, **kwargs)
                    return torch.ones_like(cos), torch.zeros_like(sin)
                return _identity_forward
            module.forward = make_identity(original_forward)
            patched.append((module, original_forward))
    try:
        yield len(patched)
    finally:
        for module, orig in patched:
            module.forward = orig


def count_rope_modules(model):
    return sum(1 for _, m in model.named_modules() if 'RotaryEmbedding' in type(m).__name__)


@contextmanager
def eager_attention(model):
    if hasattr(model, 'set_attn_implementation'):
        original = model.config._attn_implementation
        model.set_attn_implementation('eager')
        try:
            yield model
        finally:
            model.set_attn_implementation(original)
    else:
        original = model.config._attn_implementation
        model.config._attn_implementation = 'eager'
        try:
            yield model
        finally:
            model.config._attn_implementation = original
