#!/usr/bin/env python3
"""Shared helpers for the Kaggle runner, the launcher and the collector (stdlib only)."""
from __future__ import annotations

import base64
import datetime as dt
import json
import os
import pathlib
import random
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
GH_FILE_LIMIT = 95 * 1024 * 1024  # GitHub rejects files > 100 MB

TOKEN_RE = re.compile(
    r"(github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|KGAT_[A-Za-z0-9]{8,})"
)
_SECRETS = set()


def register_secret(*values):
    """Remember secret values so they are masked in every log line and output file."""
    for v in values:
        if v and len(v) >= 6:
            _SECRETS.add(v)
            _SECRETS.add(base64.b64encode(f"x-access-token:{v}".encode()).decode())


def scrub(text):
    if not isinstance(text, str):
        return text
    for s in sorted(_SECRETS, key=len, reverse=True):
        text = text.replace(s, "***")
    return TOKEN_RE.sub("***", text)


def log(*parts):
    print(scrub(" ".join(str(p) for p in parts)), flush=True)


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def iso(t=None):
    return (t or utcnow()).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_config(root=ROOT):
    return json.loads((pathlib.Path(root) / "config" / "project.json").read_text())


class GitError(RuntimeError):
    pass


class FrozenError(RuntimeError):
    pass


# --------------------------------------------------------------------------- GitHub REST
def gh_api(method, path, token, body=None, retries=3, timeout=60):
    """Returns (status, json, headers). status 0 means a network failure."""
    url = path if path.startswith("http") else "https://api.github.com" + path
    data = json.dumps(body).encode() if body is not None else None
    last = None
    for i in range(retries):
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "nid-tools",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode(errors="replace").strip()
                js = json.loads(raw) if raw[:1] in "{[" else {"raw": raw}
                return r.status, js, dict(r.headers)
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            try:
                js = json.loads(raw)
            except Exception:
                js = {"raw": raw}
            if e.code in (500, 502, 503, 504) and i < retries - 1:
                time.sleep(2 * (i + 1))
                continue
            return e.code, js, dict(e.headers)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            time.sleep(2 * (i + 1))
    return 0, {"error": str(last)}, {}


# --------------------------------------------------------------------------- git
def git(args, cwd, token=None, check=True, timeout=1800):
    cmd = ["git"]
    if token:
        b = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        cmd += ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {b}"]
        register_secret(token)
    cmd += list(args)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    p = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout)
    out = scrub((p.stdout or "") + (p.stderr or ""))
    if check and p.returncode != 0:
        raise GitError(f"git {' '.join(args[:2])} failed ({p.returncode}): {out.strip()[-800:]}")
    return p.returncode, out


def clone_repo(owner, repo, branch, token, dest, name="nid-runner", email="nid-runner@users.noreply.github.com"):
    """Shallow clone of the tip of `branch` (works for private repos when a token is given)."""
    dest = pathlib.Path(dest)
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    git(["init", "-q", "-b", branch], dest)
    base = os.environ.get("NID_GIT_BASE", "https://github.com")
    git(["remote", "add", "origin", f"{base}/{owner}/{repo}.git"], dest)
    git(["config", "user.name", name], dest)
    git(["config", "user.email", email], dest)
    git(["config", "http.postBuffer", "524288000"], dest)
    git(["config", "http.version", "HTTP/1.1"], dest)
    last = None
    for i in range(4):
        try:
            git(["fetch", "-q", "--depth", "1", "origin", branch], dest, token)
            git(["checkout", "-q", "-B", branch, "FETCH_HEAD"], dest)
            return dest
        except GitError as e:
            last = e
            time.sleep(3 * (i + 1))
    raise last


def commit_tree(repo_dir, branch, token, rel_dest, src_dir, message, must_not_exist=True, attempts=6):
    """Copy src_dir to repo_dir/rel_dest, commit and push. Rebuilds on top of the newest
    remote tip on every attempt so concurrent pushes from another lane cannot conflict.
    With must_not_exist the destination is never overwritten (frozen runs)."""
    repo_dir = pathlib.Path(repo_dir)
    last = None
    for i in range(attempts):
        try:
            git(["fetch", "-q", "--depth", "1", "origin", branch], repo_dir, token)
            git(["reset", "-q", "--hard", "FETCH_HEAD"], repo_dir)
            target = repo_dir / rel_dest
            if target.exists():
                if must_not_exist:
                    raise FrozenError(f"{rel_dest} already exists on {branch}; frozen runs are never overwritten")
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_dir, target)
            git(["add", "-A", "--", rel_dest], repo_dir)
            rc, _ = git(["diff", "--cached", "--quiet"], repo_dir, check=False)
            if rc == 0:
                raise GitError("nothing to commit")
            git(["commit", "-q", "-m", message], repo_dir)
            git(["push", "-q", "origin", f"HEAD:refs/heads/{branch}"], repo_dir, token)
            return git(["rev-parse", "HEAD"], repo_dir)[1].strip()
        except FrozenError:
            raise
        except GitError as e:
            last = e
            log(f"[git] attempt {i + 1}/{attempts} failed: {e}")
            time.sleep(min(60, 5 * (i + 1) + random.random() * 5))
    raise last


def remote_has_path(repo_dir, branch, token, rel):
    git(["fetch", "-q", "--depth", "1", "origin", branch], repo_dir, token)
    git(["reset", "-q", "--hard", "FETCH_HEAD"], repo_dir)
    return (pathlib.Path(repo_dir) / rel).exists()


def unique_dest(repo_dir, rel):
    """failed/<run>/<nb> may be attempted several times; never overwrite an earlier attempt."""
    base, n = rel, 1
    while (pathlib.Path(repo_dir) / rel).exists():
        n += 1
        rel = f"{base}-attempt{n}"
    return rel
