#!/usr/bin/env python3
"""Fast static check (stdlib only, seconds, no GPU): run before spending Kaggle GPU hours.

* every file under src/ compiles
* every `from <pkg>.<module> import name` in a notebook / module points at a name that exists
* a notebook does not both import a shared name and define it again
* every notebook listed in config/project.json exists and parses
Exit code 1 on any problem.
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
problems = []


def top_names(path):
    tree = ast.parse(path.read_text())
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in n.names}
    return names


def module_file(dotted):
    p = SRC.joinpath(*dotted.split("."))
    if p.with_suffix(".py").exists():
        return p.with_suffix(".py")
    if (p / "__init__.py").exists():
        return p / "__init__.py"
    return None


def check_imports(tree, where, pkgs, current=None):
    for n in ast.walk(tree):
        if not isinstance(n, ast.ImportFrom):
            continue
        if n.level:                                    # relative import inside the package
            if current is None:
                continue
            base = current.split(".")[:-1]
            base = base[: len(base) - (n.level - 1)] if n.level > 1 else base
            target = ".".join(base + ([n.module] if n.module else []))
        elif n.module and n.module.split(".")[0] in pkgs:
            target = n.module
        else:
            continue
        f = module_file(target)
        if f is None:
            problems.append(f"{where}: module '{target}' not found in src/")
            continue
        have = top_names(f)
        for a in n.names:
            if a.name != "*" and a.name not in have and module_file(f"{target}.{a.name}") is None:
                problems.append(f"{where}: '{a.name}' is not defined in {target}")


def main():
    pkgs = {p.name for p in SRC.iterdir() if (p / "__init__.py").exists()} if SRC.is_dir() else set()
    for f in sorted(SRC.rglob("*.py")) if SRC.is_dir() else []:
        rel = str(f.relative_to(ROOT))
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError as e:
            problems.append(f"{rel}: syntax error {e}")
            continue
        r = f.relative_to(SRC).with_suffix("")
        check_imports(tree, rel, pkgs, ".".join(r.parts))
    cfg = json.loads((ROOT / "config" / "project.json").read_text())
    for nid, nb in cfg["notebooks"].items():
        p = ROOT / nb["file"]
        if not p.exists():
            problems.append(f"{nb['file']}: listed in config/project.json but missing")
            continue
        try:
            raw = json.loads(p.read_text())
        except Exception as e:
            problems.append(f"{nb['file']}: not valid JSON ({e})")
            continue
        code = "\n".join("".join(c["source"]) for c in raw["cells"] if c["cell_type"] == "code")
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            problems.append(f"{nb['file']}: syntax error {e}")
            continue
        check_imports(tree, nb["file"], pkgs)
        imported = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
                    and n.module.split(".")[0] in pkgs for a in n.names}
        defined = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        for d in sorted(set(defined) & imported):
            problems.append(f"{nb['file']}: '{d}' is imported from the shared package AND defined in the notebook")
    if problems:
        print("repo check FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print(f"repo check OK ({len(cfg['notebooks'])} notebooks, packages: {', '.join(sorted(pkgs)) or 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
