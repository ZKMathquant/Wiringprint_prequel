"""Weight-space transformations (permutations, rotations, noise, knock-outs, fine-tuning) producing ConditionModels.
"""

import copy
import torch
from contextlib import contextmanager
from .arch import get_attn, get_head_dim, get_layers, get_mlp, get_num_heads, get_num_kv_heads
from .condition import ConditionModel
from .device import free_memory


# ---------------------------------------------------------------------------
# HPC note (fixes the PART 4 OOM from the previous run): every random-
# orthogonal-matrix construction in this notebook (local MLP rotation, global
# residual rotation, the PART 4 rotation-angle sweep) needs a `dim x dim`
# fp32 QR decomposition, and cuSOLVER's QR kernel only supports fp32/fp64 --
# so building it directly on `DEVICE` forces a full fp32 d_ff x d_ff (or
# d_model x d_model) buffer onto the single busiest, most memory-constrained
# device in the whole notebook, on top of three already-resident checkpoints.
# CPU RAM is far more abundant and is untouched by everything else here, so
# `cpu_orthogonal` builds and QR-decomposes ENTIRELY on CPU; callers move only
# the final matrix (cast down to the model's dtype, which is also half the
# fp32 size) to GPU, right before the one matmul that needs it, for as short
# a time as possible.
# ---------------------------------------------------------------------------
def cpu_orthogonal(dim, seed=0):
    """Random `dim x dim` orthogonal matrix, built and QR-decomposed on CPU.

    Returns a CPU float32 tensor. Cast + .to(device) only the matrix you still
    need, right before consuming it -- never hold the fp32 CPU copy and a GPU
    copy alive at the same time (see make_local_orthogonal_mlp,
    make_global_rotation, rotation_angle_sweep_mlp for the pattern). Uses its
    own CPU generator instead of the global RNG (torch.manual_seed) so it has
    no side effect on unrelated random draws elsewhere in the notebook.
    """
    gen = torch.Generator(device='cpu').manual_seed(seed)
    A = torch.randn(dim, dim, dtype=torch.float32, generator=gen)
    Q, _ = torch.linalg.qr(A)
    del A
    return Q


def make_aligned_mlp_permutation(model, layer_idx, seed=0, device='cpu'):
    '''Returns a ConditionModel (not a full model copy) representing `model` with
    the d_ff hidden axis of one MLP block permuted: the SAME permutation P applied
    to gate_proj's and up_proj's output rows (nn.Linear layout: weight is (out, in),
    so permuting outputs = permuting weight ROWS, y = x @ W^T + b), and down_proj's
    input columns compensated by P^T. Only this one layer's tensors are ever
    duplicated -- see the ConditionModel cell in Part 0 for why.'''
    mlp = get_mlp(model, layer_idx)
    d_ff = mlp.gate_proj.out_features
    torch.manual_seed(seed)
    perm = torch.randperm(d_ff)
    param_refs = [(mlp.gate_proj, 'weight'), (mlp.up_proj, 'weight'), (mlp.down_proj, 'weight')]
    variants = [
        mlp.gate_proj.weight.data[perm, :].clone(),
        mlp.up_proj.weight.data[perm, :].clone(),
        mlp.down_proj.weight.data[:, perm].clone(),
    ]
    if mlp.gate_proj.bias is not None:
        param_refs.append((mlp.gate_proj, 'bias')); variants.append(mlp.gate_proj.bias.data[perm].clone())
    if mlp.up_proj.bias is not None:
        param_refs.append((mlp.up_proj, 'bias')); variants.append(mlp.up_proj.bias.data[perm].clone())
    cm = ConditionModel(model, param_refs, variants, label=f'aligned_mlp_L{layer_idx}')
    return cm, perm


def make_unaligned_mlp_permutation(model, layer_idx, seed=0, device='cpu'):
    '''Same permutation applied to gate_proj/up_proj, but down_proj is left
    UNCOMPENSATED -- the broken-symmetry negative control. ConditionModel-based,
    same memory profile as the aligned case above.'''
    mlp = get_mlp(model, layer_idx)
    d_ff = mlp.gate_proj.out_features
    torch.manual_seed(seed)
    perm = torch.randperm(d_ff)
    param_refs = [(mlp.gate_proj, 'weight'), (mlp.up_proj, 'weight')]
    variants = [
        mlp.gate_proj.weight.data[perm, :].clone(),
        mlp.up_proj.weight.data[perm, :].clone(),
    ]
    if mlp.gate_proj.bias is not None:
        param_refs.append((mlp.gate_proj, 'bias')); variants.append(mlp.gate_proj.bias.data[perm].clone())
    if mlp.up_proj.bias is not None:
        param_refs.append((mlp.up_proj, 'bias')); variants.append(mlp.up_proj.bias.data[perm].clone())
    # down_proj intentionally left untouched -- not included in param_refs at all
    cm = ConditionModel(model, param_refs, variants, label=f'unaligned_mlp_L{layer_idx}')
    return cm, perm


def make_gateonly_mlp_permutation(model, layer_idx, seed=0, device='cpu'):
    '''Architecture-specific third condition with no GPT-2 analogue: permute ONLY
    gate_proj's output, leaving up_proj and down_proj untouched. Tests whether the
    two SwiGLU/GeGLU branches are independently load-bearing.'''
    mlp = get_mlp(model, layer_idx)
    d_ff = mlp.gate_proj.out_features
    torch.manual_seed(seed)
    perm = torch.randperm(d_ff)
    param_refs = [(mlp.gate_proj, 'weight')]
    variants = [mlp.gate_proj.weight.data[perm, :].clone()]
    if mlp.gate_proj.bias is not None:
        param_refs.append((mlp.gate_proj, 'bias')); variants.append(mlp.gate_proj.bias.data[perm].clone())
    cm = ConditionModel(model, param_refs, variants, label=f'gateonly_mlp_L{layer_idx}')
    return cm, perm


def build_block_permutation(n_blocks, block_dim, perm, device='cpu', dtype=torch.float32):
    '''Block-permutation matrix over `n_blocks` blocks of size `block_dim` each.
    `dtype` MUST match the tensors it will be matrix-multiplied against (e.g. the
    model's bf16/fp16 weights) -- torch.matmul requires both operands to share a
    dtype, and this defaulted to float32 while the model runs in bf16/fp16, which
    is exactly what produced the "expected mat1 and mat2 to have the same dtype"
    RuntimeError below.'''
    total = n_blocks * block_dim
    P = torch.zeros(total, total, device=device, dtype=dtype)
    for new_pos, old_block in enumerate(perm):
        rows = torch.arange(old_block * block_dim, (old_block + 1) * block_dim)
        cols = torch.arange(new_pos * block_dim, (new_pos + 1) * block_dim)
        P[rows, cols] = 1.0
    return P


def permute_attention_heads_gqa(model, layer_idx=0, mode='aligned', seed=0, device='cpu'):
    '''mode in {'aligned', 'unaligned_no_compensation', 'unaligned_within_group'}.
    Returns a ConditionModel (per the memory-safety note in Part 0) -- only this
    layer's Q/K/V/O tensors are ever duplicated, never the whole model.

    aligned: permute KV-head groups (group = num_heads // num_kv_heads adjacent query
      heads sharing one KV head); Q heads move with their parent group; K/V heads permuted
      the same way; o_proj input compensated by P^T. Exact symmetry.

    unaligned_no_compensation: same Q/K/V permutation as aligned, but o_proj left
      uncompensated (direct GQA analogue of the original notebook's negative control).

    unaligned_within_group: permutes individual QUERY heads across DIFFERENT groups
      (i.e. reassigns a query head to a different KV parent) -- has no GPT-2 analogue,
      since GPT-2's group size is 1. Always breaks the symmetry even with o_proj
      "compensated" for the group-level permutation, because query heads no longer
      attend using the K/V they were trained against.
    '''
    attn = get_attn(model, layer_idx)
    n_heads, n_kv = get_num_heads(model), get_num_kv_heads(model)
    head_dim = get_head_dim(model)
    group_size = n_heads // n_kv

    attn_dtype = attn.q_proj.weight.dtype
    torch.manual_seed(seed)
    group_perm = torch.randperm(n_kv).tolist()   # permutation over KV-head groups
    P_kv = build_block_permutation(n_kv, head_dim, group_perm, device=device, dtype=attn_dtype)
    q_head_perm = []
    for g in group_perm:
        q_head_perm.extend(range(g * group_size, (g + 1) * group_size))
    P_q = build_block_permutation(n_heads, head_dim, q_head_perm, device=device, dtype=attn_dtype)

    param_refs, variants = [], []

    def add(module, attr, new_tensor):
        param_refs.append((module, attr)); variants.append(new_tensor)

    if mode in ('aligned', 'unaligned_no_compensation'):
        add(attn.q_proj, 'weight', P_q.T @ attn.q_proj.weight.data)
        add(attn.k_proj, 'weight', P_kv.T @ attn.k_proj.weight.data)
        add(attn.v_proj, 'weight', P_kv.T @ attn.v_proj.weight.data)
        if attn.q_proj.bias is not None:
            add(attn.q_proj, 'bias', P_q.T @ attn.q_proj.bias.data)
        if attn.k_proj.bias is not None:
            add(attn.k_proj, 'bias', P_kv.T @ attn.k_proj.bias.data)
        if attn.v_proj.bias is not None:
            add(attn.v_proj, 'bias', P_kv.T @ attn.v_proj.bias.data)
        if mode == 'aligned':
            add(attn.o_proj, 'weight', attn.o_proj.weight.data @ P_q)
        # else: o_proj left uncompensated on purpose (not added to param_refs)

    elif mode == 'unaligned_within_group':
        within_group_perm = torch.randperm(n_heads).tolist()  # ignores group structure
        P_wg = build_block_permutation(n_heads, head_dim, within_group_perm, device=device, dtype=attn_dtype)
        add(attn.q_proj, 'weight', P_wg.T @ attn.q_proj.weight.data)
        if attn.q_proj.bias is not None:
            add(attn.q_proj, 'bias', P_wg.T @ attn.q_proj.bias.data)
        add(attn.o_proj, 'weight', attn.o_proj.weight.data @ P_wg)
        # K/V left untouched -- Q heads now paired with the wrong KV parent

    return ConditionModel(model, param_refs, variants, label=f'{mode}_heads_L{layer_idx}')


def make_local_orthogonal_mlp(model, layer_idx, seed=0, device='cpu'):
    '''ConditionModel version: rotates one MLP block's d_ff axis by a random
    orthogonal Q. Only this layer's tensors are duplicated.'''
    mlp = get_mlp(model, layer_idx)
    d_ff = mlp.gate_proj.out_features
    # Q is built and QR-decomposed on CPU (see cpu_orthogonal) -- this GPU
    # never has to hold a full fp32 d_ff x d_ff buffer, only the final
    # model-dtype Q used in the two small matmuls below. (Old approach built
    # + QR'd A directly on `device`; cuSOLVER's QR only supports fp32/fp64
    # anyway, so that bought nothing but VRAM pressure.)
    Q = cpu_orthogonal(d_ff, seed=seed).to(device=device, dtype=mlp.gate_proj.weight.dtype)
    param_refs = [(mlp.gate_proj, 'weight'), (mlp.up_proj, 'weight'), (mlp.down_proj, 'weight')]
    variants = [
        Q.T @ mlp.gate_proj.weight.data,
        Q.T @ mlp.up_proj.weight.data,
        mlp.down_proj.weight.data @ Q,
    ]
    return ConditionModel(model, param_refs, variants, label=f'local_rotation_L{layer_idx}')


def make_global_rotation(model, seed=0, device='cpu'):
    '''Rotates the whole residual stream by a single orthogonal Q. NOTE: because
    RMSNorm's per-channel learned scale does not commute with a generic rotation
    (unlike a hypothetical scale-free normalization), this is expected to be only
    approximately, not exactly, identity-preserving -- unlike the aligned-permutation
    cases above. The numeric check quantifies exactly how far off it is.

    Unlike the single-layer perturbations above, this genuinely touches (almost)
    every tensor in the model, so the ConditionModel it returns is not "cheap" the
    way the others are -- but it's still only ONE transient set of full-size variant
    tensors, created and freed one at a time (see the loop below), rather than the
    naive pattern of holding many complete deep-copied models simultaneously.'''
    d_model = model.config.hidden_size
    # Same fix as make_local_orthogonal_mlp above: build + QR on CPU, move
    # only the final model-dtype matrix to GPU.
    Q = cpu_orthogonal(d_model, seed=seed).to(device=device, dtype=model.model.embed_tokens.weight.dtype)

    param_refs, variants = [], []
    def add(module, attr, new_tensor):
        param_refs.append((module, attr)); variants.append(new_tensor)

    add(model.model.embed_tokens, 'weight', model.model.embed_tokens.weight.data @ Q)
    for layer in get_layers(model):
        add(layer.self_attn.q_proj, 'weight', layer.self_attn.q_proj.weight.data @ Q)
        add(layer.self_attn.k_proj, 'weight', layer.self_attn.k_proj.weight.data @ Q)
        add(layer.self_attn.v_proj, 'weight', layer.self_attn.v_proj.weight.data @ Q)
        add(layer.self_attn.o_proj, 'weight', Q.T @ layer.self_attn.o_proj.weight.data)
        add(layer.mlp.gate_proj, 'weight', layer.mlp.gate_proj.weight.data @ Q)
        add(layer.mlp.up_proj, 'weight', layer.mlp.up_proj.weight.data @ Q)
        add(layer.mlp.down_proj, 'weight', Q.T @ layer.mlp.down_proj.weight.data)
    if hasattr(model, 'lm_head') and not getattr(model.config, 'tie_word_embeddings', False):
        add(model.lm_head, 'weight', model.lm_head.weight.data @ Q)

    return ConditionModel(model, param_refs, variants, label='global_rotation')


def make_gaussian_noise_mlp(model, layer_idx, sigma_frac=0.1, seed=0, device='cpu'):
    mlp = get_mlp(model, layer_idx)
    torch.manual_seed(seed)
    param_refs = [(mlp.gate_proj, 'weight'), (mlp.up_proj, 'weight'), (mlp.down_proj, 'weight')]
    variants = []
    for (module, attr) in param_refs:
        w = getattr(module, attr).data
        sigma = sigma_frac * w.std()
        variants.append(w + torch.randn_like(w) * sigma)
    return ConditionModel(model, param_refs, variants, label=f'gaussian_noise_L{layer_idx}_s{sigma_frac}')


def make_neuron_deletion_mlp(model, layer_idx, delete_frac=0.1, seed=0, device='cpu'):
    mlp = get_mlp(model, layer_idx)
    d_ff = mlp.gate_proj.out_features
    torch.manual_seed(seed)
    n_delete = int(round(d_ff * delete_frac))
    delete_idx = torch.randperm(d_ff)[:n_delete]

    gate_w = mlp.gate_proj.weight.data.clone(); gate_w[delete_idx, :] = 0.0
    up_w = mlp.up_proj.weight.data.clone(); up_w[delete_idx, :] = 0.0
    down_w = mlp.down_proj.weight.data.clone(); down_w[:, delete_idx] = 0.0
    param_refs = [(mlp.gate_proj, 'weight'), (mlp.up_proj, 'weight'), (mlp.down_proj, 'weight')]
    variants = [gate_w, up_w, down_w]
    if mlp.gate_proj.bias is not None:
        b = mlp.gate_proj.bias.data.clone(); b[delete_idx] = 0.0
        param_refs.append((mlp.gate_proj, 'bias')); variants.append(b)
    if mlp.up_proj.bias is not None:
        b = mlp.up_proj.bias.data.clone(); b[delete_idx] = 0.0
        param_refs.append((mlp.up_proj, 'bias')); variants.append(b)
    return ConditionModel(model, param_refs, variants, label=f'neuron_deletion_L{layer_idx}_f{delete_frac}')


def partial_violation_mlp_permutation(model, layer_idx, violation_frac, seed=0, device='cpu'):
    '''ConditionModel version (see Part 0): this sweep calls the function
    7 fracs x 5 seeds = 35 times per model, which under a naive deepcopy-per-call
    pattern would mean 35 full-model duplications per model, back to back. Here
    each call only ever allocates one layer's worth of replacement tensors.'''
    mlp = get_mlp(model, layer_idx)
    d_ff = mlp.gate_proj.out_features
    torch.manual_seed(seed)

    perm = torch.randperm(d_ff)
    inverse_perm = torch.argsort(perm)
    comp_perm = inverse_perm.clone()
    n_violate = int(round(d_ff * violation_frac))
    if n_violate >= 2:
        violate_positions = torch.randperm(d_ff)[:n_violate]
        shuffled_values = comp_perm[violate_positions][torch.randperm(n_violate)]
        comp_perm[violate_positions] = shuffled_values

    param_refs = [(mlp.gate_proj, 'weight'), (mlp.up_proj, 'weight'), (mlp.down_proj, 'weight')]
    variants = [
        mlp.gate_proj.weight.data[perm, :].clone(),
        mlp.up_proj.weight.data[perm, :].clone(),
        mlp.down_proj.weight.data[:, comp_perm].clone(),
    ]
    if mlp.gate_proj.bias is not None:
        param_refs.append((mlp.gate_proj, 'bias')); variants.append(mlp.gate_proj.bias.data[perm].clone())
    if mlp.up_proj.bias is not None:
        param_refs.append((mlp.up_proj, 'bias')); variants.append(mlp.up_proj.bias.data[perm].clone())
    return ConditionModel(model, param_refs, variants, label=f'partial_violation_L{layer_idx}_v{violation_frac}')


def make_aligned_mlp_permutation_real(model, layer_idx, seed=0, device='cpu'):
    '''Like make_aligned_mlp_permutation (Part 2), but bakes the permutation into
    REAL, persisted weights on a full model copy instead of a transient
    ConditionModel swap. This is required for training: ConditionModel's
    swap-forward-restore pattern sets `.data = variant` for the forward pass and
    restores `.data = original` in a `finally` block that runs BEFORE backward()
    is called from the training loop. backward() itself still computes gradients
    correctly with respect to the variant that was actually used in forward (this
    was checked empirically -- PyTorch's saved-tensors are a snapshot, not a live
    read through `.data`), but by the time `opt.step()` runs, `.data` has already
    been reset to the ORIGINAL weight. That means the update becomes
    `original_weight -= lr * grad_of_permuted_variant`: a permuted-order gradient
    landing on unpermuted-order storage, which is a coordinate-basis mismatch, not
    a valid SGD step. Baking the permutation in for real avoids this: the copy's
    weights and its own gradients live in the SAME (permuted) basis throughout
    training, so ordinary SGD is valid, and the aligned copy is a genuine
    standalone model at eval time (no stale/frozen swap-in required).

    Same permutation convention as make_aligned_mlp_permutation: gate_proj and
    up_proj output rows permuted together, down_proj input columns compensated by
    the same permutation, so the function computed by the MLP is unchanged before
    any fine-tuning happens.

    Memory note: this moves `model` to CPU before deep-copying it, so the copy is made
    CPU-to-CPU rather than leaving the original and its full copy briefly co-resident on
    GPU (a transient ~2x spike that's enough to OOM a ~2.6B-parameter model on a 15GB
    card once combined with the training copy's own gradient buffers a few lines below).
    `model` is left on CPU when this returns -- move it back to `device` yourself if you
    still need it there (the call site below doesn't; it evicts `model` right after
    calling this anyway).
    '''
    model.to('cpu')                      # avoid a transient 2x-on-GPU spike during deepcopy
    m = copy.deepcopy(model)
    mlp = get_mlp(m, layer_idx)
    torch.manual_seed(seed)
    perm = torch.randperm(mlp.gate_proj.out_features)
    with torch.no_grad():
        mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[perm, :].clone()
        mlp.up_proj.weight.data = mlp.up_proj.weight.data[perm, :].clone()
        mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, perm].clone()
        if mlp.gate_proj.bias is not None:
            mlp.gate_proj.bias.data = mlp.gate_proj.bias.data[perm].clone()
        if mlp.up_proj.bias is not None:
            mlp.up_proj.bias.data = mlp.up_proj.bias.data[perm].clone()
        # down_proj.bias (if present) lives on the OUTPUT dim (hidden_size), not
        # d_ff, so it's untouched by this permutation -- same as the ConditionModel
        # version in Part 2.
    # BUG FIX (confirmed by a real Kaggle run): `device` was accepted as a parameter but
    # never actually used inside this function -- the deep copy `m` inherits CPU placement
    # from `model` (moved to CPU a few lines above) and was returned as-is, still on CPU.
    # The call site then ran `finetune_in_place(aligned_before_ft, ..., device=DEVICE)`,
    # which only moves the training text's token ids to DEVICE, not the model itself, so
    # the embedding lookup crashed with "Expected all tensors to be on the same device,
    # but got index is on cuda:0, different from other tensors on cpu" the instant training
    # started. The baseline branch a few lines below avoids this by chaining `.to(DEVICE)`
    # onto its own deepcopy explicitly; this function needs the same fix applied
    # internally, since every caller passes device=DEVICE expecting it to be honored.
    m = m.to(device)
    return m, perm


def finetune_in_place(model, tokenizer, texts, n_steps=5, lr=1e-5, device='cpu'):
    '''Trains `model` in place with ordinary SGD and returns it. Never wraps a
    ConditionModel (see make_aligned_mlp_permutation_real's docstring for why
    that's unsafe to combine with backward()+step()). The caller owns the
    model's lifetime: create your copy right before calling this, and
    `del` it + call free_memory() as soon as you're done comparing it, so
    at most one extra full-model copy is ever alive at a time -- important
    on a 16GB T4, where the three base checkpoints alone already use most of
    the fp16 budget (see PART 3, STEP C's memory note below).'''
    for p in model.parameters():
        p.requires_grad_(True)  # base models are loaded with requires_grad_(False)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for step in range(n_steps):
        opt.zero_grad()
        for text in texts:
            ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
            if ids.shape[1] < 2:
                continue
            out = model(ids, labels=ids)
            out.loss.backward()
        opt.step()
    opt.zero_grad(set_to_none=True)  # drop the last step's .grad buffers now
    model.eval()                     # rather than waiting on del+gc to reclaim them
    for p in model.parameters():
        p.requires_grad_(False)
    return model


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

    # SPEED FIX (confirmed by two real Kaggle runs: the first completed but took 4.8 hours and
    # crashed elsewhere; the second was cancelled by the user at 4.6 hours for taking too long).
    # This is the single largest cost in the whole notebook. The per-theta QR below was CPU-only
    # -- deliberately, to fix a real GPU OOM from an even earlier version (see the module-level
    # comment above cpu_orthogonal). But that OOM came from TWO factors together: building
    # several full d_ff x d_ff buffers on GPU simultaneously, AND materializing every theta's
    # ConditionModel up front. The one-at-a-time generator design below already fixes the second
    # factor, which means only ONE d_ff x d_ff fp32 buffer (~250-340MB for these three models'
    # d_ff) needs to be alive on GPU at any instant -- trivial against the ~3.5GB headroom this
    # notebook already runs with, three models resident. cuSOLVER's GPU QR (geqrf/orgqr) is
    # roughly one to two orders of magnitude faster than CPU LAPACK at this matrix size, and this
    # function is called ~90 times per model by Sec 8.4's 15-seed sweep alone, plus 5 more times
    # PER LAYER by dilution_probe_batched (Sec 8.6's full-depth M2 sweep calls that once per
    # layer) -- so this single change is the highest-leverage fix available for total runtime.
    # Falls back to the original CPU path automatically if the GPU buffer can't be allocated,
    # so a genuinely memory-constrained run degrades to "slow but correct" rather than crashing.
    try:
        Q_full_dev = Q_full.to(device=device) if device != 'cpu' else Q_full
        I_dev = I.to(device=device) if device != 'cpu' else I
    except torch.cuda.OutOfMemoryError:
        Q_full_dev, I_dev = Q_full, I
        free_memory()

    for theta in angles:
        # interpolate rotation generator in the Lie algebra sense: exponential map of
        # theta * logm(Q_full) would be ideal, but for a lightweight sweep we linearly
        # blend Q_full and I then re-orthogonalize -- a simple, inspectable proxy for
        # "how much rotation." Blend + re-QR run on `qr_device` (GPU when available and
        # successfully allocated, else CPU); only the resulting matrix, already cast to
        # model_dtype (half the fp32 footprint), is what actually gets used below.
        try:
            blend = (1 - theta) * I_dev + theta * Q_full_dev
            Qb_raw, _ = torch.linalg.qr(blend)
            del blend
        except torch.cuda.OutOfMemoryError:
            # Extremely defensive: even the single transient buffer didn't fit this time
            # (e.g. another process briefly spiked usage). Retry this one theta on CPU
            # rather than aborting the whole sweep.
            free_memory()
            blend_cpu = (1 - theta) * I + theta * Q_full
            Qb_raw, _ = torch.linalg.qr(blend_cpu)
            del blend_cpu
        Qb = Qb_raw.to(device=device, dtype=model_dtype)
        del Qb_raw
        mlp = get_mlp(model, layer_idx)
        param_refs = [(mlp.gate_proj, 'weight'), (mlp.up_proj, 'weight'), (mlp.down_proj, 'weight')]
        variants = [
            Qb.T @ mlp.gate_proj.weight.data,
            Qb.T @ mlp.up_proj.weight.data,
            mlp.down_proj.weight.data @ Qb,
        ]
        del Qb
        yield theta, ConditionModel(model, param_refs, variants, label=f'rotation_theta{theta}_L{layer_idx}')


def head_knockout_mask(num_heads, head_dim, keep_fraction, seed=0):
    gen = torch.Generator().manual_seed(seed)
    n_keep = max(1, int(round(num_heads * keep_fraction)))
    keep_heads = torch.randperm(num_heads, generator=gen)[:n_keep]
    mask = torch.zeros(num_heads, dtype=torch.bool)
    mask[keep_heads] = True
    return mask.repeat_interleave(head_dim)


@contextmanager
def head_knockout(attn_module, num_heads, head_dim, keep_fraction, seed=0):
    """Real causal ablation: zeros a (1 - keep_fraction) share of the REAL o_proj input's
    head-slices via a forward pre-hook on the real, trained o_proj."""
    if keep_fraction >= 1.0:
        yield
        return
    mask = head_knockout_mask(num_heads, head_dim, keep_fraction, seed=seed)
    def _hook(module, inputs):
        x = inputs[0]
        m = mask.to(device=x.device)
        return (x * m.to(x.dtype),) + inputs[1:]
    handle = attn_module.o_proj.register_forward_pre_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def make_neuron_knockout(model, layer_idx, knockout_idx, device='cpu'):
    """ConditionModel: zeros the listed d_ff-axis neurons' contribution by zeroing the
    corresponding COLUMNS of the real down_proj weight -- removes exactly what that neuron adds
    to the residual stream, the cleanest single-tensor causal ablation available here."""
    mlp = get_mlp(model, layer_idx)
    W = mlp.down_proj.weight.data.clone()
    W[:, knockout_idx] = 0.0
    return ConditionModel(model, [(mlp.down_proj, 'weight')], [W], label=f'knockout{len(knockout_idx)}_L{layer_idx}')


def make_rank_truncated_downproj(model, layer_idx, rank, device='cpu'):
    """ConditionModel: replaces the REAL down_proj weight with its own best rank-`rank`
    approximation (truncated SVD reconstruction of the actual trained matrix), leaving
    gate_proj/up_proj untouched."""
    mlp = get_mlp(model, layer_idx)
    W = mlp.down_proj.weight.data.float().to('cpu')
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = min(rank, S.shape[0])
    W_trunc = (U[:, :r] * S[:r]) @ Vh[:r, :]
    W_trunc = W_trunc.to(device=device, dtype=mlp.down_proj.weight.dtype)
    return ConditionModel(model, [(mlp.down_proj, 'weight')], [W_trunc], label=f'rank{rank}_L{layer_idx}')
