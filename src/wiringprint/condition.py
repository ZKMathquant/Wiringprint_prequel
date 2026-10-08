"""ConditionModel: a cheap, reversible 'perturbed model' view that swaps parameter tensors in place.
"""

import torch
from contextlib import contextmanager
from .arch import get_mlp


# ---------------------------------------------------------------------------
# ConditionModel: a cheap "view" of an already-loaded model with a small set
# of tensors swapped to a stored variant, instead of a full deep copy.
# ---------------------------------------------------------------------------
class ConditionModel:
    """Represents `base_model` with the tensors in `param_refs` temporarily
    replaced by `variant_tensors` during each forward call.

    param_refs:      list of (module, attr_name) pairs, e.g. (mlp.gate_proj, 'weight')
    variant_tensors: list of replacement tensors, same order/shape as param_refs

    Only ever allocates the (small, single-layer-sized) variant tensors --
    never a second copy of the whole model. Swaps are applied right before the
    wrapped forward call and undone right after (even on exception), so the
    shared base model always holds its original weights between calls.
    """
    def __init__(self, base_model, param_refs, variant_tensors, label=""):
        assert len(param_refs) == len(variant_tensors)
        self.base_model = base_model
        self.param_refs = param_refs
        self.variant_tensors = variant_tensors
        self.label = label

    def __call__(self, *args, **kwargs):
        saved = [getattr(mod, attr).data for (mod, attr) in self.param_refs]
        for (mod, attr), variant in zip(self.param_refs, self.variant_tensors):
            getattr(mod, attr).data = variant
        try:
            return self.base_model(*args, **kwargs)
        finally:
            for (mod, attr), orig in zip(self.param_refs, saved):
                getattr(mod, attr).data = orig

    def __getattr__(self, name):
        # Forward everything else (.config, .eval(), .model, register_forward_hook
        # on submodules via get_layers(), etc.) straight to the shared base model.
        return getattr(self.base_model, name)

    def __repr__(self):
        return f"ConditionModel(label={self.label!r}, n_swapped_tensors={len(self.param_refs)})"


def _smoke_test_condition_model(model, tokenizer, device):
    mlp = get_mlp(model, 0)
    ids = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        base_logits = model(ids).logits.clone()
    zeroed = torch.zeros_like(mlp.down_proj.weight.data)
    cm = ConditionModel(model, [(mlp.down_proj, 'weight')], [zeroed], label='smoke_test_zero_down_proj')
    with torch.no_grad():
        perturbed_logits = cm(ids).logits
        restored_logits = model(ids).logits  # weights should already be back to normal here
    diff_perturbed = (base_logits - perturbed_logits).abs().max().item()
    diff_restored = (base_logits - restored_logits).abs().max().item()
    ok = diff_perturbed > 1e-3 and diff_restored < 1e-4
    return ok, diff_perturbed, diff_restored


@contextmanager
def bypass_module(module):
    """Temporarily replace `module.forward` with the identity function, then restore the
    original -- a real causal ablation of a specific real submodule's real, learned weights."""
    original_forward = module.forward
    module.forward = lambda x: x
    try:
        yield module
    finally:
        module.forward = original_forward


def check_scaling_symmetry(model, layer_idx, alpha=2.0, seed=0, device='cpu'):
    '''Applies W1 -> alpha*W1, W2 -> W2/alpha to gate_proj/down_proj and measures
    the logit deviation. For a homogeneous activation this would be exactly 0;
    for SiLU/GeGLU it should be nonzero, growing with |alpha - 1|. ConditionModel
    version -- only gate_proj/down_proj's tensors are duplicated.'''
    mlp = get_mlp(model, layer_idx)
    param_refs = [(mlp.gate_proj, 'weight'), (mlp.down_proj, 'weight')]
    variants = [mlp.gate_proj.weight.data * alpha, mlp.down_proj.weight.data / alpha]
    if mlp.gate_proj.bias is not None:
        param_refs.append((mlp.gate_proj, 'bias'))
        variants.append(mlp.gate_proj.bias.data * alpha)
    return ConditionModel(model, param_refs, variants, label=f'scaling_alpha{alpha}_L{layer_idx}')
