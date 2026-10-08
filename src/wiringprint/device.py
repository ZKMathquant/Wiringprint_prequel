"""Device, dtype and GPU-memory helpers (OOM guard, autocast context, pinned/async transfers).

Notebook variables read live (set them in the notebook, after `wp.bind(globals())`):
    DEVICE, DTYPE, PRECISION_MODE, _EST_TOTAL_PARAMS
"""

import contextlib
import gc
import torch
from contextlib import nullcontext
from .runtime import ambient as _nb, LATE as _LATE


def pick_dtype():
    if _nb.DEVICE != 'cuda':
        return torch.float32, 'float32 (CPU run)'
    if _nb.PRECISION_MODE == 'fp32':
        return torch.float32, 'float32 (forced)'
    if _nb.PRECISION_MODE == 'bf16':
        return torch.bfloat16, 'bfloat16 (forced)'
    if _nb.PRECISION_MODE == 'fp16':
        return torch.float16, 'float16 (forced)'
    total_gpu = torch.cuda.get_device_properties(0).total_memory
    fp32_bytes = _nb._EST_TOTAL_PARAMS * 4 * 1.3  # +30% headroom for activations/temp tensors
    if fp32_bytes < total_gpu * 0.85:
        return torch.float32, 'float32 (auto: fp32 fits comfortably on this GPU)'
    bf16_ok = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    name = 'bfloat16' if bf16_ok else 'float16'
    return dtype, f"{name} (auto: fp32 would not fit all three checkpoints on this GPU)"


# ---------------------------------------------------------------------------
# Memory-safety helpers, used throughout the notebook after anything that
# touches GPU memory in a non-trivial way (model loads, whole-model rotations,
# fine-tuning copies). Cheap per-layer perturbations (see ConditionModel,
# defined right after MODEL_CONFIGS below) mostly don't need these, since they
# never allocate a whole extra model in the first place.
# ---------------------------------------------------------------------------
def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def print_gpu_mem(tag=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        label = f" [{tag}]" if tag else ""
        print(f"  GPU mem{label}: allocated={alloc:.2f}GB  reserved={reserved:.2f}GB  total={total:.2f}GB")


@contextlib.contextmanager
def cuda_oom_guard(label=""):
    """Wrap a GPU-memory-heavy block so an OOM prints actionable diagnostics
    and frees the cache before propagating -- instead of a bare traceback
    that papermill/Kaggle turns into a full from-scratch re-run. Does NOT
    swallow the error: an OOM here means a real memory bug upstream (an extra
    full-size buffer landed on GPU, or a condition wasn't freed before the
    next one was built) that should be fixed at the source, not silently
    retried into a hang.
    """
    try:
        yield
    except torch.cuda.OutOfMemoryError:
        print(f"\n  !! CUDA OOM in {label or '(unlabeled block)'} !!")
        print_gpu_mem("at OOM")
        free_memory()
        print_gpu_mem("after emergency free_memory()")
        raise


# ---------------------------------------------------------------------------
# Pinned-memory batched inputs. This is a transfer optimization, not a task change.
# ---------------------------------------------------------------------------
def tokenize_pinned(tokenizer, texts, device=_LATE):
    if device is _LATE: device = _nb.DEVICE
    enc = tokenizer(
        list(texts), padding=True, truncation=False, return_tensors="pt",
        pad_to_multiple_of=8
    )
    out = {}
    for k, v in enc.items():
        v = v.contiguous()
        if device == "cuda":
            v = v.pin_memory()
        out[k] = v
    return out


def to_device_async(batch, device=_LATE):
    if device is _LATE: device = _nb.DEVICE
    if device != "cuda":
        return {k: v.to(device) for k, v in batch.items()}
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _amp_ctx():
    if _nb.DEVICE == 'cuda':
        return torch.autocast(device_type='cuda', dtype=_nb.DTYPE)
    return nullcontext()
