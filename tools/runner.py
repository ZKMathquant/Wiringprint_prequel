#!/usr/bin/env python3
"""Kaggle-side runner. Executes ONE notebook with papermill, then commits the result to GitHub.

Called from the generated kernel stub (see launch.py). Success -> runs/<run_id>/<notebook>/
(frozen, never overwritten). Failure/timeout -> failed/<run_id>/<notebook>/ with the partial
notebook, log and any files produced so far. A copy of the bundle is always left in
/kaggle/working/bundle so `collect.yml` can recover it if the push itself failed.
"""
from __future__ import annotations

import json
import os
import pathlib
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import nid_common as C  # noqa: E402
from nid_common import log  # noqa: E402

SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git", "wandb", "hf_cache", ".cache", "huggingface"}
TEXT_SUFFIXES = {".json", ".csv", ".tsv", ".txt", ".log", ".md", ".py", ".yaml", ".yml", ".ipynb", ".html"}


class PreflightError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- checks
def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception:
        return []


def http_status(url, token=None, method="HEAD", timeout=30):
    headers = {"User-Agent": "nid-runner"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method=method, headers=headers), timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def hf_checks(models, hf_token, needs_hf):
    out = []
    if not hf_token:
        out.append(("FAIL" if needs_hf else "WARN", "hf_token", "HF_TOKEN not available (Kaggle Secret / inline)"))
        return out
    st = http_status("https://huggingface.co/api/whoami-v2", hf_token, "GET")
    if st != 200:
        out.append(("FAIL", "hf_token", f"HF token rejected by huggingface.co (HTTP {st})"))
        return out
    out.append(("OK", "hf_token", "HF token is valid"))
    for m in models:
        st = http_status(f"https://huggingface.co/{m}/resolve/main/config.json", hf_token)
        if st == 200:
            out.append(("OK", f"hf_model:{m}", "accessible"))
        elif st == 403:
            out.append(("FAIL", f"hf_model:{m}", "gated - accept the licence on the model page with the token's account"))
        else:
            out.append(("FAIL", f"hf_model:{m}", f"not reachable (HTTP {st})"))
    return out


def github_checks(owner, repo, gh_token):
    if not gh_token:
        return [("FAIL", "gh_token", "GH_TOKEN not available (Kaggle Secret / inline)")]
    st, js, _ = C.gh_api("GET", f"/repos/{owner}/{repo}", gh_token)
    if st != 200:
        return [("FAIL", "gh_token", f"cannot read {owner}/{repo} (HTTP {st}): {js.get('message', '')}")]
    if not js.get("permissions", {}).get("push", False):
        return [("FAIL", "gh_token", "token can read but NOT push to the repo (needs Contents: read & write)")]
    res = [("OK", "gh_token", f"push access to {owner}/{repo}")]
    st, _, _ = C.gh_api("GET", f"/repos/{owner}/{repo}/actions/workflows/run.yml", gh_token)
    res.append(("OK" if st == 200 else "WARN", "gh_actions",
                "can read workflows (chaining needs Actions: read & write)" if st == 200
                else f"cannot read run.yml (HTTP {st}); chaining 'all' will not work"))
    return res


def check_gpu(cfg, hard):
    gpus = gpu_info()
    need = cfg.get("require_gpu_substring") or ""
    if not gpus:
        return [("FAIL" if hard else "INFO", "gpu", "no GPU visible to nvidia-smi")]
    if need and not any(need.lower() in g.lower() for g in gpus):
        return [("FAIL" if hard else "WARN", "gpu", f"expected '{need}' but got {gpus}")]
    return [("OK", "gpu", "; ".join(gpus))]


def report(results, tag="preflight"):
    for level, name, msg in results:
        log(f"[{tag}] {level:4s} {name}: {msg}")


def check_package(root):
    """Import every module of src/<package> on the real Kaggle stack (CPU only, no GPU quota needed)."""
    import importlib
    src = pathlib.Path(root) / "src"
    if not src.is_dir():
        return [("INFO", "src", "no src/ directory in this repo")]
    sys.path.insert(0, str(src))
    out = []
    for pkg in sorted(p.name for p in src.iterdir() if (p / "__init__.py").exists()):
        for f in sorted((src / pkg).rglob("*.py")):
            rel = f.relative_to(src).with_suffix("")
            mod = ".".join(rel.parts[:-1] if rel.name == "__init__" else rel.parts)
            try:
                importlib.import_module(mod)
                out.append(("OK", f"import:{mod}", "imports"))
            except Exception as e:
                out.append(("FAIL", f"import:{mod}", f"{type(e).__name__}: {str(e)[:200]}"))
    return out


def selfcheck(RUN, cfg, root):
    gh, hf = os.environ.get("GH_TOKEN"), os.environ.get("HF_TOKEN")
    models = sorted({m for nb in cfg["notebooks"].values() for m in nb.get("hf_models", [])})
    needs_hf = any(nb.get("needs_hf") for nb in cfg["notebooks"].values())
    res = [("INFO", "python", platform.python_version())]
    res += [("OK" if http_status(u) in (200, 301, 302) else "FAIL", f"internet:{u}", "reachable")
            for u in ("https://github.com", "https://huggingface.co")]
    res += github_checks(cfg["owner"], cfg["repo"], gh)
    res += hf_checks(models, hf, needs_hf)
    res += check_gpu(cfg, hard=False)
    res += check_package(root)
    report(res, "selfcheck")
    fails = [r for r in res if r[0] == "FAIL"]
    pathlib.Path("/kaggle/working").mkdir(parents=True, exist_ok=True)
    pathlib.Path("/kaggle/working/selfcheck.json").write_text(json.dumps(res, indent=2))
    log(f"[selfcheck] RESULT: {'FAIL' if fails else 'PASS'} ({len(fails)} failing checks)")
    return 1 if fails else 0


# ----------------------------------------------------------------------------- execution
def ensure_papermill():
    try:
        import papermill  # noqa: F401
        return
    except ImportError:
        pass
    for i in range(3):
        rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "papermill"]).returncode
        if rc == 0:
            return
        time.sleep(5 * (i + 1))
    raise PreflightError("could not pip install papermill")


def snapshot(d):
    snap = {}
    for p in pathlib.Path(d).rglob("*"):
        if p.is_file():
            st = p.stat()
            snap[str(p.relative_to(d))] = (st.st_size, st.st_mtime_ns)
    return snap


def execute(src, dst, cwd, env, log_path, budget_s):
    """Run papermill, stream its log, enforce the wall-clock budget. Returns (rc, timed_out)."""
    cmd = [sys.executable, "-m", "papermill", str(src), str(dst), "--cwd", str(cwd),
           "-k", "python3", "--no-progress-bar", "--log-output", "--log-level", "INFO"]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=True, errors="replace")
    cap = 30 * 1024 * 1024

    def pump():
        written = 0
        with open(log_path, "w", errors="replace") as fh:
            for line in proc.stdout:
                line = C.scrub(line)
                print(line, end="", flush=True)
                if written < cap:
                    fh.write(line)
                    written += len(line)
                elif written == cap:
                    fh.write("\n[log truncated at 30 MB]\n")
                    written += 1

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    timed_out = False
    try:
        proc.wait(timeout=max(budget_s, 1))
    except subprocess.TimeoutExpired:
        timed_out = True
        log(f"[runner] wall-clock budget exhausted ({budget_s:.0f}s) - stopping notebook to save partial results")
        for sig, grace in ((signal.SIGTERM, 45), (signal.SIGKILL, 10)):
            try:
                os.killpg(proc.pid, sig)
                proc.wait(timeout=grace)
                break
            except (subprocess.TimeoutExpired, ProcessLookupError):
                continue
    t.join(timeout=30)
    return (proc.returncode if proc.returncode is not None else -9), timed_out


def src_tree_hash(src_dir):
    """Stable fingerprint of the shared code a run used (recorded in run_manifest.json)."""
    import hashlib
    h = hashlib.sha256()
    d = pathlib.Path(src_dir)
    if not d.is_dir():
        return None
    for p in sorted(d.rglob("*.py")):
        h.update(str(p.relative_to(d)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def notebook_stats(path):
    try:
        nb = json.loads(pathlib.Path(path).read_text())
    except Exception:
        return {}
    cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    done = sum(1 for c in cells if c.get("execution_count") is not None)
    err = None
    for i, c in enumerate(nb.get("cells", [])):
        for o in c.get("outputs", []) or []:
            if o.get("output_type") == "error":
                err = {"cell_index": i, "ename": o.get("ename"), "evalue": C.scrub(str(o.get("evalue")))[:500]}
                break
        if err:
            break
    return {"code_cells": len(cells), "code_cells_executed": done, "first_error": err}


def scrub_files(root):
    for p in pathlib.Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES and p.stat().st_size <= 50 * 1024 * 1024:
            try:
                txt = p.read_text(errors="strict")
            except Exception:
                continue
            new = C.scrub(txt)
            if new != txt:
                p.write_text(new)


def build_bundle(bundle, nb_id, executed, log_path, workdir, before, manifest, max_mb):
    shutil.rmtree(bundle, ignore_errors=True)
    out = bundle / "outputs"
    out.mkdir(parents=True)
    skipped = []
    if executed.exists():
        shutil.copy2(executed, bundle / f"{nb_id}.executed.ipynb")
    if log_path.exists():
        shutil.copy2(log_path, bundle / "run.log")
    try:
        frz = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120).stdout
        (bundle / "pip_freeze.txt").write_text(frz)
    except Exception:
        pass
    (bundle / "gpu.txt").write_text("\n".join(gpu_info()))
    after = snapshot(workdir)
    new = [(rel, meta[0]) for rel, meta in after.items() if before.get(rel) != meta]
    new = [(rel, size) for rel, size in new if not (set(pathlib.PurePath(rel).parts) & SKIP_DIRS)]
    new.sort(key=lambda t: t[1])
    total = 0
    for rel, size in new:
        if size > C.GH_FILE_LIMIT:
            skipped.append({"file": rel, "bytes": size, "reason": "over GitHub 95 MB file limit"})
        elif total + size > max_mb * 1024 * 1024:
            skipped.append({"file": rel, "bytes": size, "reason": f"bundle cap {max_mb} MB reached"})
        else:
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pathlib.Path(workdir) / rel, dst)
            total += size
    manifest["output_files"] = sum(1 for _ in out.rglob("*") if _.is_file())
    manifest["output_bytes"] = total
    manifest["skipped_files"] = skipped
    scrub_files(bundle)
    (bundle / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return bundle


# ----------------------------------------------------------------------------- chaining
def dispatch_next(RUN, status, gh_token):
    rest = RUN.get("queue_rest") or []
    if not rest:
        return "no further notebooks"
    if status != "success" and RUN.get("on_failure", "continue") == "stop":
        return "stopped: on_failure=stop"
    inputs = {"notebooks": ",".join(rest), "run_id": RUN["run_id"], "slot": RUN["slot"], "lanes": "1",
              "on_failure": RUN.get("on_failure", "continue"), "label": "", "resume_from": "",
              "selfcheck": "false", "pin_sha": RUN["sha"]}
    last = ""
    for i in range(3):
        st, js, _ = C.gh_api("POST", f"/repos/{RUN['owner']}/{RUN['repo']}/actions/workflows/run.yml/dispatches",
                             gh_token, {"ref": RUN["branch"], "inputs": inputs})
        if st == 204:
            return f"dispatched next: {','.join(rest)}"
        last = f"HTTP {st}: {js.get('message', js)}"
        time.sleep(5 * (i + 1))
    return "CHAIN_ERROR " + last


# ----------------------------------------------------------------------------- main
def main(RUN, root):
    root = pathlib.Path(root)
    cfg = C.load_config(root)
    gh, hf = os.environ.get("GH_TOKEN", ""), os.environ.get("HF_TOKEN", "")
    C.register_secret(gh, hf)
    if RUN.get("mode") == "selfcheck":
        return selfcheck(RUN, cfg, root)

    t0 = time.time()
    nb_id, run_id = RUN["notebook"], RUN["run_id"]
    owner, repo, branch = RUN["owner"], RUN["repo"], RUN["branch"]
    work_root = pathlib.Path(os.environ.get("NID_WORK_ROOT", "/tmp/nid_work"))
    out_root = pathlib.Path(os.environ.get("NID_OUT_ROOT", "/kaggle/working/bundle"))
    workdir = work_root / nb_id
    executed = work_root / f"{nb_id}.executed.ipynb"
    log_path = work_root / f"{nb_id}.run.log"
    shutil.rmtree(work_root, ignore_errors=True)
    workdir.mkdir(parents=True)

    manifest = {"run_id": run_id, "notebook": nb_id, "repo_sha": RUN["sha"], "slot": RUN["slot"],
                "started_utc": C.iso(), "python": platform.python_version(), "status": "runner_error",
                "reason": "", "resume_from": RUN.get("resume_from") or None, "chain": ""}
    push_repo = work_root / "push_repo"
    repo_ok = False
    status, reason, before = "runner_error", "", {}
    try:
        nb_cfg = cfg["notebooks"].get(nb_id)
        if not nb_cfg:
            raise PreflightError(f"notebook '{nb_id}' is not in config/project.json at commit {RUN['sha'][:7]}")
        res = github_checks(owner, repo, gh) + check_gpu(cfg, hard=True)
        res += hf_checks(nb_cfg.get("hf_models", []), hf, nb_cfg.get("needs_hf", False)) if (nb_cfg.get("needs_hf") or hf) else []
        report(res)
        fails = [r for r in res if r[0] == "FAIL"]
        if any(r[1] == "gh_token" for r in fails):
            raise PreflightError("; ".join(f"{r[1]}: {r[2]}" for r in fails))
        ga = cfg.get("git_author") or {}
        C.clone_repo(owner, repo, branch, gh, push_repo,
                     **({"name": ga["name"], "email": ga["email"]} if ga.get("name") and ga.get("email") else {}))
        repo_ok = True
        frozen_rel = f"runs/{run_id}/{nb_id}"
        if (push_repo / frozen_rel).exists():
            status, reason = "refused_frozen", f"{frozen_rel} already exists - frozen runs are never overwritten"
            raise C.FrozenError(reason)
        if fails:
            raise PreflightError("; ".join(f"{r[1]}: {r[2]}" for r in fails))
        # Notebook AND shared code come from the pinned checkout (`root`, the exact launch commit), so they
        # can never be from different commits even if the branch moved while the run sat in Kaggle's queue.
        src_nb = root / nb_cfg["file"]
        if not src_nb.exists():
            raise PreflightError(f"{nb_cfg['file']} missing in repo at the pinned commit")
        src_dir = root / "src"
        manifest["src_sha256"] = src_tree_hash(src_dir)
        shutil.copy2(src_nb, workdir / f"{nb_id}.ipynb")
        before = snapshot(workdir)  # only the input notebook is excluded; seeded files are carried forward
        resume = RUN.get("resume_from") or ""
        if resume:
            rp = pathlib.PurePosixPath(resume)
            if ".." in rp.parts or rp.parts[:1] not in (("failed",), ("runs",)):
                raise PreflightError(f"resume_from must live under failed/ or runs/: {resume}")
            seed = push_repo / resume / "outputs"
            if not seed.is_dir():
                raise PreflightError(f"resume_from has no outputs/ directory: {resume}")
            shutil.copytree(seed, workdir, dirs_exist_ok=True)
            log(f"[runner] seeded working dir from {resume}/outputs")
        ensure_papermill()

        env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": os.pathsep.join(p for p in (str(src_dir), os.environ.get("PYTHONPATH", "")) if p),
               "HF_HOME": os.environ.get("HF_HOME", "/tmp/hf"), "NID_RUN_ID": run_id, "NID_NOTEBOOK": nb_id}
        if hf:
            env["HF_TOKEN"] = env["HUGGING_FACE_HUB_TOKEN"] = hf
        env.pop("GH_TOKEN", None)  # the notebook never needs the GitHub token
        limit = cfg.get("session_limit_hours", 12) * 3600 - cfg.get("safety_margin_minutes", 40) * 60
        budget = limit - (time.time() - t0)
        log(f"[runner] executing {nb_id} (budget {budget / 3600:.2f} h)")
        rc, timed_out = execute(workdir / f"{nb_id}.ipynb", executed, workdir, env, log_path, budget)
        if timed_out:
            status, reason = "timeout", "wall-clock budget exhausted before the notebook finished"
        elif rc == 0 and executed.exists():
            status = "success"
        else:
            status, reason = "failed", f"papermill exit code {rc}"
    except C.FrozenError:
        pass
    except PreflightError as e:
        status, reason = "preflight_failed", str(e)
        log(f"[runner] PREFLIGHT FAILED: {reason}")
    except Exception:
        status, reason = "runner_error", traceback.format_exc()[-3000:]
        log(f"[runner] unexpected error:\n{reason}")

    manifest.update(status=status, reason=C.scrub(reason), finished_utc=C.iso(),
                    duration_s=round(time.time() - t0), **notebook_stats(executed))

    pushed = None
    if status != "refused_frozen":
        bundle = build_bundle(out_root / nb_id, nb_id, executed, log_path, workdir, before, manifest,
                              cfg.get("max_bundle_mb", 400))
        if repo_ok:
            try:
                base = "runs" if status == "success" else "failed"
                rel = f"{base}/{run_id}/{nb_id}"
                if status != "success":
                    rel = C.unique_dest(push_repo, rel)
                pushed = C.commit_tree(push_repo, branch, gh, rel, bundle,
                                       f"{base[:-1] if base == 'runs' else 'failed'} {run_id}: {nb_id} [{status}]",
                                       must_not_exist=True)
                manifest["pushed_to"], manifest["commit"] = rel, pushed
                log(f"[runner] pushed {rel} -> commit {pushed[:8]}")
            except C.FrozenError as e:
                log(f"[runner] {e}")
                status = "refused_frozen"
            except Exception as e:
                log(f"[runner] PUSH FAILED: {e}. Bundle kept in {bundle}; use the 'collect' workflow to recover it.")
                manifest["push_error"] = str(e)[:500]
        else:
            log(f"[runner] no GitHub access; bundle kept in {bundle} (use the 'collect' workflow)")

    chain = ""
    if RUN.get("queue_rest") and status != "refused_frozen":
        chain = dispatch_next(RUN, status, gh)
        log(f"[runner] chain: {chain}")
        if chain.startswith("CHAIN_ERROR") and repo_ok:
            tmp = pathlib.Path(tempfile.mkdtemp())
            (tmp / "error.txt").write_text(
                f"{C.iso()}  could not start the next notebooks {RUN['queue_rest']} after {nb_id}.\n{chain}\n"
                f"Start them manually: Actions -> Run notebooks on Kaggle -> notebooks={','.join(RUN['queue_rest'])} run_id={run_id}\n")
            try:
                C.commit_tree(push_repo, branch, gh, f"runs/{run_id}/_chain_errors/{nb_id}", tmp, f"chain error after {nb_id}")
            except Exception as e:
                log(f"[runner] could not record chain error: {e}")
    manifest["chain"] = chain
    log(f"[runner] DONE {nb_id}: {status} in {manifest['duration_s']}s")
    return 0 if status == "success" and not manifest.get("push_error") else 1
