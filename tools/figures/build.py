#!/usr/bin/env python
"""Render and check the documentation figures.

    python tools/figures/build.py                    # render every registered figure
    python tools/figures/build.py --only strategies  # one page (or several)
    python tools/figures/build.py --only research --name dsr-null
    python tools/figures/build.py --list             # what is registered, what is rendered
    python tools/figures/build.py --check            # PNGs, image links and embeds are consistent
    python tools/figures/build.py --jobs 4           # render in 4 processes
    python tools/figures/build.py --synthetic --out /tmp/preview   # fresh clone preview

Figures are ``@register``-ed ``draw(pal)`` functions in ``tools/figures/fig_*.py`` (see
tools/figures/README.md). Each renders to ``docs/_assets/figures/<page>/<name>-{light,dark}.png``
with :func:`aurum.viz.style.save_figure`. A figure whose ``needs`` are missing is skipped and
its committed PNGs are kept; byte-identical output is not rewritten. Exit status: 0 OK,
1 a render or check failed, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("MPLBACKEND", "Agg")

import _common as C  # noqa: E402
import _docscan as D  # noqa: E402


# ---------------------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------------------
def module_names() -> list[str]:
    return sorted(p.stem for p in HERE.glob("fig_*.py"))


def discover(names: list[str] | None = None) -> tuple[dict[str, C.FigureSpec], list[str]]:
    """Import ``fig_*.py`` modules under the import guard; return (registry, errors)."""
    errors: list[str] = []
    for mod in names if names is not None else module_names():
        try:
            with C.import_guard():
                importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001 - one broken module must not hide the rest
            errors.append(f"{mod}.py failed to import: {type(exc).__name__}: {exc}")
    return dict(C.REGISTRY), errors


def select(registry: dict[str, C.FigureSpec], pages: list[str] | None, names: list[str] | None) -> list[C.FigureSpec]:
    specs = sorted(registry.values(), key=lambda s: (s.page, s.name))
    if pages:
        specs = [s for s in specs if s.page in pages]
    if names:
        specs = [s for s in specs if s.name in names]
    return specs


# ---------------------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------------------
def _render_in_worker(module: str, key: str, out_root: str | None, synthetic: bool) -> C.RenderResult:
    for p in (str(REPO), str(HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    C.set_force_synthetic(synthetic)
    with C.import_guard():
        importlib.import_module(module)
    return C.render(C.REGISTRY[key], out_root=Path(out_root) if out_root else None, preview=synthetic)


def _print_result(r: C.RenderResult) -> None:
    tag = {"updated": "ok  ", "unchanged": "same", "skipped": "skip", "error": "FAIL"}[r.status]
    extra = f"  {r.seconds:5.1f}s" if r.status in ("updated", "unchanged", "error") else "        "
    print(f"[{tag}] {r.spec.key:<48}{extra}  {r.message.splitlines()[0] if r.message else r.status}")
    if r.status == "error" and "\n" in r.message:
        for line in r.message.splitlines()[1:]:
            print(f"         {line}")
    for n in r.notes:
        print(f"         note: {n}")


def cmd_render(specs: list[C.FigureSpec], *, jobs: int, out_root: Path | None, synthetic: bool) -> int:
    if not specs:
        print("no figures selected (see --list)")
        return 0
    t0 = time.perf_counter()
    results: list[C.RenderResult] = []
    if jobs > 1 and len(specs) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_render_in_worker, s.module, s.key, str(out_root) if out_root else None,
                              synthetic): s for s in specs}
            for f in as_completed(futs):
                try:
                    r = f.result()
                except Exception as exc:  # noqa: BLE001
                    r = C.RenderResult(futs[f], "error", message=f"{type(exc).__name__}: {exc}")
                results.append(r)
                _print_result(r)
    else:
        for s in specs:
            r = C.render(s, out_root=out_root, preview=synthetic)
            results.append(r)
            _print_result(r)
    counts = {k: sum(r.status == k for r in results) for k in ("updated", "unchanged", "skipped", "error")}
    print(f"\n{len(results)} figures in {time.perf_counter() - t0:.1f}s: {counts['updated']} updated, "
          f"{counts['unchanged']} unchanged, {counts['skipped']} skipped, {counts['error']} failed")
    return 1 if counts["error"] else 0


# ---------------------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------------------
def cmd_list(specs: list[C.FigureSpec]) -> int:
    if not specs:
        print("no figures registered (add tools/figures/fig_<page>.py; see tools/figures/README.md)")
        return 0
    print(f"{'figure':<48} {'pngs':<8} {'needs':<44} doc")
    for s in specs:
        light, dark = s.paths()
        pngs = "both" if light.exists() and dark.exists() else ("partial" if light.exists() or dark.exists()
                                                                else "missing")
        miss = set(s.missing_needs())
        needs = ", ".join(f"{n}{' (missing)' if n in miss else ''}" for n in s.needs) or "-"
        print(f"{s.key:<48} {pngs:<8} {needs:<44} {s.doc}")
    print(f"\n{len(specs)} figures in {len({s.module for s in specs})} modules")
    return 0


# ---------------------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------------------
def _in_scope(path: Path, pages: list[str] | None) -> bool:
    if not pages:
        return True
    try:
        parts = path.resolve().relative_to(C.FIGURES_DIR).parts
    except ValueError:
        return False
    return bool(parts) and parts[0] in pages


def check(registry: dict[str, C.FigureSpec], pages: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Return (errors, infos). Scoped to ``pages`` (figure dirs) when given."""
    errors: list[str] = []
    infos: list[str] = []
    specs = select(registry, pages, None)

    # 1. every registered figure has both PNGs
    for s in specs:
        for p in s.paths():
            if not p.exists():
                errors.append(f"{s.key}: {D.rel(p)} not rendered (run build.py --only {s.page})")

    # 2. every local image referenced from docs/*.md and README.md exists; embeds are well formed
    referenced: set[Path] = set()
    for f in D.doc_files():
        for r in D.refs(f):
            if r.kind != "image":
                continue
            target = r.resolve()
            if target is None:
                continue
            referenced.add(target)
            if not _in_scope(target, pages):
                continue
            if not target.exists():
                errors.append(f"{r.where()}: image {r.target} does not exist")
        for pic in D.pictures(f):
            paths = [(f.parent / x).resolve() for x in (pic.light, pic.dark) if x]
            if pages and not any(_in_scope(p, pages) for p in paths):
                continue
            for prob in pic.problems:
                errors.append(f"{pic.where()}: <picture> {prob}")

    # 3. every PNG under docs/_assets/figures belongs to a registered figure and is referenced
    known = {p.resolve() for s in registry.values() for p in s.paths()}
    for png in sorted(C.FIGURES_DIR.rglob("*.png")) if C.FIGURES_DIR.exists() else []:
        if not _in_scope(png, pages):
            continue
        if png.resolve() not in known:
            errors.append(f"{D.rel(png)}: not produced by any registered figure (cannot be rebuilt); "
                          "register it or delete it")
        if png.resolve() not in referenced:
            errors.append(f"{D.rel(png)}: orphan (not referenced from docs/*.md or README.md)")
    n_png = sum(1 for p in C.FIGURES_DIR.rglob("*.png") if _in_scope(p, pages)) if C.FIGURES_DIR.exists() else 0
    infos.append(f"{len(specs)} registered figures, {n_png} PNGs under {D.rel(C.FIGURES_DIR)}"
                 + (f" (pages: {', '.join(pages)})" if pages else ""))
    return errors, infos


def cmd_check(registry: dict[str, C.FigureSpec], import_errors: list[str], pages: list[str] | None) -> int:
    errors, infos = check(registry, pages)
    errors = import_errors + errors
    for line in infos:
        print(line)
    if errors:
        print(f"\n{len(errors)} problem(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("check OK")
    return 0


# ---------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tools/figures/build.py", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", metavar="PAGE", help="only these pages (figure directories)")
    p.add_argument("--name", nargs="+", metavar="NAME", help="only figures with these names")
    p.add_argument("--list", action="store_true", help="list registered figures and their PNG status")
    p.add_argument("--check", action="store_true",
                   help="verify PNGs, image references, <picture> embeds and orphans (no rendering)")
    p.add_argument("--jobs", type=int, default=1, metavar="N", help="render in N processes (default 1)")
    p.add_argument("--out", type=Path, metavar="DIR",
                   help="render into DIR/<page>/ instead of docs/_assets/figures (preview)")
    p.add_argument("--synthetic", action="store_true",
                   help="force the synthetic data fallback everywhere (preview on a clone without "
                        "data_store/; requires --out)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.synthetic and (args.out is None or args.out.resolve() == C.FIGURES_DIR.resolve()):
        print("--synthetic needs --out DIR outside docs/_assets/figures "
              "(synthetic renders must never replace the committed real-data PNGs)", file=sys.stderr)
        return 2
    if args.jobs < 1:
        print("--jobs must be >= 1", file=sys.stderr)
        return 2
    C.set_force_synthetic(args.synthetic)
    registry, import_errors = discover()
    if args.only:
        unknown = sorted(set(args.only) - {s.page for s in registry.values()})
        if unknown and not args.check:
            print(f"note: no registered figures for page(s) {', '.join(unknown)}")
    if args.check:
        return cmd_check(registry, import_errors, args.only)
    for e in import_errors:
        print(f"[FAIL] {e}")
    specs = select(registry, args.only, args.name)
    if args.list:
        rc = cmd_list(specs)
        return 1 if import_errors else rc
    rc = cmd_render(specs, jobs=args.jobs, out_root=args.out, synthetic=args.synthetic)
    return 1 if import_errors else rc


if __name__ == "__main__":
    sys.exit(main())
