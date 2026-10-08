"""Performance infrastructure: torch.compile, attention backend choice, generation-cache benchmark, on-disk result cache.

Notebook variables read live (set them in the notebook, after `wp.bind(globals())`):
    ATTN_IMPL, CACHE_DIR, COMPILE_ENABLED, COMPILE_MODE, DEVICE, EVAL_CORPUS, GPU_HEADROOM_FRAC
"""

import importlib.util
import json
import time
import torch
import warnings
from .runtime import ambient as _nb, LATE as _LATE
from .arch import get_layers, get_mlp
from .device import free_memory
from .stats import stable_hash


# ---------------------------------------------------------------------------
# Backend selection: use the fastest *actually installed and compatible* path.
# ---------------------------------------------------------------------------
def _has_pkg(name):
    return importlib.util.find_spec(name) is not None


def choose_attention_backend():
    candidates = []
    if _has_pkg("flash_attn"):
        candidates.append("flash_attention_2")
    if _has_pkg("xformers"):
        candidates.append("xformers")
    candidates.append("eager")
    return candidates[0], candidates


def apply_attention_backend(model, backend=_LATE):
    """Apply only a backend that the installed Transformers version exposes."""
    if backend is _LATE: backend = _nb.ATTN_IMPL
    if backend == "eager":
        return False
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation(backend)
            return True
        except Exception as exc:
            warnings.warn(f"{backend} unavailable for {model.config.model_type}: {type(exc).__name__}: {exc}. Using eager.")
    return False


def cache_key(**kwargs):
    return stable_hash(kwargs)


def cache_get(key):
    p = _nb.CACHE_DIR / f"{key}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def cache_put(key, obj):
    (_nb.CACHE_DIR / f"{key}.json").write_text(json.dumps(obj, sort_keys=True))


def try_compile_model(name, model, tokenizer):
    if _nb.DEVICE != "cuda" or not hasattr(torch, "compile") or not _nb.COMPILE_ENABLED:
        return model, False, "compile unavailable/disabled"
    before = torch.cuda.memory_reserved()
    try:
        probe = tokenizer(_nb.EVAL_CORPUS[:2], return_tensors="pt", padding=True)
        probe = {k: v.to(_nb.DEVICE) for k, v in probe.items()}
        compiled = torch.compile(model, mode=_nb.COMPILE_MODE, dynamic=False, fullgraph=False)
        with torch.inference_mode():
            _ = compiled(**probe).logits[:, -1, :10]
        torch.cuda.synchronize()

        after = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        if after > total * _nb.GPU_HEADROOM_FRAC:
            # Compilation is an optimization, never a license to consume the memory
            # needed by the actual scientific conditions.
            del compiled
            free_memory()
            return model, False, "disabled: compiled cache exceeded VRAM headroom"

        # Mutation-consistency probe: if a live weight change is not reflected in the
        # compiled graph, using it for perturbation experiments would be scientifically invalid.
        layer = get_mlp(model, min(0, len(get_layers(model))-1))
        w = layer.down_proj.weight
        original = w.data
        mutated = original.clone()
        mutated.view(-1)[0] += torch.tensor(0.125, device=mutated.device, dtype=mutated.dtype)
        with torch.inference_mode():
            base_out = compiled(**probe).logits[:, -1, :10].float()
            w.data = mutated
            mut_out = compiled(**probe).logits[:, -1, :10].float()
            w.data = original
        changed = not torch.allclose(base_out, mut_out, rtol=0.0, atol=1e-7)
        if not changed:
            del compiled
            free_memory()
            return model, False, "disabled: compile mutation-consistency probe failed"
        return compiled, True, f"compiled; compile VRAM delta={(after-before)/1e9:.2f} GB"
    except Exception as exc:
        warnings.warn(f"torch.compile failed for {name}: {type(exc).__name__}: {exc}")
        free_memory()
        return model, False, f"fallback:{type(exc).__name__}"


def benchmark_generation_cache(model, tok, prompt, new_tokens=16):
    # BUG FIX (confirmed by a real Kaggle run: this cell is the last thing that ran before
    # "Kernel died while waiting for execute reply" -- no Python traceback, just dynamo
    # recompile-limit warnings immediately before). Root cause: every model was torch.compile'd
    # in Part 0A; calling generate(cache_implementation="offloaded_static") on a compiled model
    # forces torch._dynamo to trace cache_utils.py's per-layer prefetch/offload .to(device) calls,
    # which are genuine CUDA-graph-breaking device copies -- each distinct layer_idx becomes its
    # own recompilation guard, blowing past config.recompile_limit (28 layers vs. a limit of 8)
    # and destabilizing the process. torch.compiler.disable() forces this benchmark to run in
    # real eager mode regardless of the model's compiled state, which is what "genuine
    # autoregressive cache benchmark" (Sec 8.11A's own stated goal) actually requires anyway --
    # benchmarking a cache implementation's real behavior, not torch.compile's interaction with it.
    model.eval()
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(_nb.DEVICE) for k,v in inputs.items()}
    attempts = ["offloaded_static", "static", None]
    rows=[]
    for impl in attempts:
        kwargs = dict(max_new_tokens=new_tokens, do_sample=False, use_cache=True)
        if impl is not None:
            kwargs["cache_implementation"] = impl
        t0=time.perf_counter()
        try:
            with torch.inference_mode(), torch.compiler.disable():
                out = model.generate(**inputs, **kwargs)
            if _nb.DEVICE=="cuda": torch.cuda.synchronize()
            rows.append({"cache_implementation":impl or "default",
                         "seconds":time.perf_counter()-t0,
                         "new_tokens":int(out.shape[1]-inputs["input_ids"].shape[1]),
                         "status":"accepted"})
            break
        except Exception as exc:
            rows.append({"cache_implementation":impl or "default",
                         "seconds":time.perf_counter()-t0,
                         "new_tokens":0,
                         "status":f"rejected:{type(exc).__name__}"})
        finally:
            free_memory()
    return rows
