#!/usr/bin/env python3
"""Recovery: pull the bundle a Kaggle kernel left in /kaggle/working/bundle and commit it.

Use when the runner finished but could not push (GitHub outage, expired token, ...).
Only the LATEST version of a kernel keeps its output on Kaggle, so collect before re-launching.
Frozen rule still applies: an existing runs/<run_id>/<notebook> is never overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import nid_common as C  # noqa: E402
import launch as L  # noqa: E402
from nid_common import log  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", default="a")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--notebook", required=True)
    a = ap.parse_args()
    cfg = C.load_config()
    L.setup_env(cfg)
    slug = L.slot_slug(cfg, a.slot)
    out = pathlib.Path(tempfile.mkdtemp(prefix="ko_"))
    rc, txt = L.kaggle(["kernels", "output", slug, "-p", str(out)])
    log(txt.strip()[-500:])
    bundle = out / "bundle" / a.notebook
    mf = bundle / "run_manifest.json"
    if not mf.exists():
        sys.exit(f"No bundle for '{a.notebook}' in the latest output of {slug}. Found: "
                 f"{[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()][:20]}")
    m = json.loads(mf.read_text())
    if m.get("run_id") != a.run_id or m.get("notebook") != a.notebook:
        sys.exit(f"Latest output belongs to run {m.get('run_id')}/{m.get('notebook')}, not {a.run_id}/{a.notebook}.")
    m["recovered_by_collect"] = True
    mf.write_text(json.dumps(m, indent=2))
    token = os.environ.get("GH_TOKEN")
    repo_dir = C.ROOT
    owner, repo = os.environ["GITHUB_REPOSITORY"].split("/", 1)
    branch = os.environ.get("GITHUB_REF_NAME", cfg.get("branch", "main"))
    work = pathlib.Path(tempfile.mkdtemp(prefix="repo_"))
    C.clone_repo(owner, repo, branch, token, work, name="github-actions[bot]",
                 email="41898282+github-actions[bot]@users.noreply.github.com")
    base = "runs" if m.get("status") == "success" else "failed"
    rel = f"{base}/{a.run_id}/{a.notebook}"
    if base == "failed":
        rel = C.unique_dest(work, rel)
    sha = C.commit_tree(work, branch, token, rel, bundle, f"recover {a.run_id}: {a.notebook} [{m.get('status')}]")
    L.summary(f"Recovered `{rel}` (status `{m.get('status')}`) in commit `{sha[:8]}`")


if __name__ == "__main__":
    main()
