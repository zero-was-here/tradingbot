"""Documentation assets stay consistent (offline, fast, no data_store/ needed).

* every image referenced from docs/*.md and README.md exists, and every ``<picture>`` embed
  follows the canonical theme-aware format (tools/figures/README.md);
* every relative link from docs/*.md and README.md resolves (with its ``#anchor`` when the
  target is a Markdown page);
* the figure modules in tools/figures import without loading data, every registered figure
  has both PNGs committed, and every PNG under docs/_assets/figures is registered and used;
* the MkDocs hook turns embeds into Material's theme-switching images and keeps currency
  dollars out of the math, and the figure loaders never reach the 2025+ holdout.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIG_TOOLS = REPO / "tools" / "figures"
if str(FIG_TOOLS) not in sys.path:
    sys.path.insert(0, str(FIG_TOOLS))

import _common as C  # noqa: E402
import _docscan as D  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def build():
    return _load(FIG_TOOLS / "build.py", "aurum_docs_figures_build")


_LOADERS = ("_real_bars", "_synthetic_bars", "_real_macro", "_synthetic_macro")


@pytest.fixture(scope="module")
def discovered(build):
    for name in _LOADERS:
        getattr(C, name).cache_clear()
    C.reset_usage()
    registry, errors = build.discover()
    touched = sorted(C.usage()) + [n for n in _LOADERS if getattr(C, n).cache_info().currsize]
    return registry, errors, touched


# ---------------------------------------------------------------------------------------
# (a) images and embeds
# ---------------------------------------------------------------------------------------
def test_image_references_exist():
    missing = []
    for f in D.doc_files():
        for r in D.refs(f):
            if r.kind == "image" and r.resolve() is not None and not r.resolve().exists():
                missing.append(f"{r.where()}: {r.target}")
    assert not missing, "missing images:\n" + "\n".join(missing)


def test_picture_embeds_are_well_formed():
    problems = []
    for f in D.doc_files():
        for pic in D.pictures(f):
            problems += [f"{pic.where()}: {p}" for p in pic.problems]
    assert not problems, "malformed <picture> embeds:\n" + "\n".join(problems)


# ---------------------------------------------------------------------------------------
# (b) links and anchors
# ---------------------------------------------------------------------------------------
def test_relative_links_resolve():
    broken = []
    for f in D.doc_files():
        for r in D.refs(f):
            if r.kind != "link":
                continue
            target = r.resolve() or f.resolve()          # "#anchor" alone -> this page
            if not target.exists():
                broken.append(f"{r.where()}: {r.target} (no such file)")
                continue
            if r.fragment and target.suffix == ".md" and r.fragment not in D.anchors(target):
                broken.append(f"{r.where()}: {r.target} (no heading anchor #{r.fragment} in {D.rel(target)})")
    assert not broken, "broken links:\n" + "\n".join(broken)


# ---------------------------------------------------------------------------------------
# (c) figure modules and committed PNGs
# ---------------------------------------------------------------------------------------
def test_figure_modules_import_without_loading_data(discovered):
    registry, errors, touched = discovered
    assert not errors, "\n".join(errors)
    assert not touched, f"data loaded while importing figure modules: {touched}"
    for spec in registry.values():
        assert spec.module.startswith("fig_"), f"{spec.key} registered outside a fig_*.py module ({spec.module})"


def test_registered_figures_have_both_pngs(discovered):
    registry, _, _ = discovered
    missing = [f"{s.key}: {D.rel(p)}" for s in registry.values() for p in s.paths() if not p.exists()]
    assert not missing, ("figures registered but not rendered (python tools/figures/build.py --only PAGE):\n"
                         + "\n".join(missing))


def test_every_figure_png_is_registered_and_referenced(build, discovered):
    registry, _, _ = discovered
    errors, _ = build.check(registry)
    errors = [e for e in errors if "not produced by any registered figure" in e or "orphan" in e]
    assert not errors, "\n".join(errors)


# ---------------------------------------------------------------------------------------
# framework guarantees
# ---------------------------------------------------------------------------------------
def test_mkdocs_hook_converts_picture_and_protects_currency():
    hook = _load(REPO / "docs" / "_hooks" / "mkdocs_hooks.py", "aurum_docs_hooks_under_test")
    md = [
        "Inline $x_t$ stays math; costs $5 and $10, or $5-$10, stay text.",
        "",
        "<picture>",
        '  <source media="(prefers-color-scheme: dark)" srcset="_assets/figures/p/x-dark.png">',
        '  <img src="_assets/figures/p/x-light.png" alt="An [example] figure">',
        "</picture>",
        "",
        "*What to look at, with $\\sigma_t$. Data: synthetic.*",
        "",
        "`$HOME` is code.",
    ]
    out = "\n".join(hook._convert(md, "", "https://github.com/o/r", "main"))
    assert "Inline $x_t$ stays math; costs \\$5 and \\$10, or \\$5-\\$10, stay text." in out
    assert "![An \\[example\\] figure](_assets/figures/p/x-light.png#only-light){ loading=lazy }" in out
    assert "![An \\[example\\] figure](_assets/figures/p/x-dark.png#only-dark){ loading=lazy }" in out
    assert '<figcaption markdown="span">*What to look at, with $\\sigma_t$. Data: synthetic.*</figcaption>' in out
    assert "<picture>" not in out and "`$HOME` is code." in out
    nested = "\n".join(hook._convert(["> [!NOTE]", ">"] + ["> " + x for x in md[2:8]], "", "", "main"))
    assert "<figure" not in nested and "    ![An \\[example\\] figure](_assets/figures/p/x-dark.png#only-dark)" in nested


def test_mkdocs_hook_rewrites_outside_links_in_figure_captions():
    """A caption link that leaves docs/ becomes a GitHub URL, like any other line (mkdocs --strict
    would otherwise fail on ``../tools/...``); links inside docs/ are left to MkDocs."""
    hook = _load(REPO / "docs" / "_hooks" / "mkdocs_hooks.py", "aurum_docs_hooks_under_test")
    repo = "https://github.com/o/r"
    block = [
        "<picture>",
        '  <source media="(prefers-color-scheme: dark)" srcset="_assets/figures/p/x-dark.png">',
        '  <img src="_assets/figures/p/x-light.png" alt="A figure">',
        "</picture>",
        "",
        "*Costs $5 ([figure code](../tools/figures/fig_x.py#L10), [run](../runs/), [results](RESULTS.md#top)). "
        "Data: synthetic.*",
    ]
    code_url = f"{repo}/blob/v2/tools/figures/fig_x.py#L10"
    out = "\n".join(hook._convert(block, "", repo, "v2"))
    assert (f'<figcaption markdown="span">*Costs \\$5 ([figure code]({code_url}), [run]({repo}/tree/v2/runs), '
            "[results](RESULTS.md#top)). Data: synthetic.*</figcaption>") in out
    assert "](../" not in out
    # a page one level down (docs/sub/page.md) climbs two levels to leave docs/
    sub = "\n".join(hook._convert([*block[:5], "*See [figure code](../../tools/figures/fig_x.py).*"], "sub", repo, "main"))
    assert f"[figure code]({repo}/blob/main/tools/figures/fig_x.py)" in sub
    # emitted as plain paragraphs inside an alert: the caption is rewritten there too
    nested = "\n".join(hook._convert(["> [!NOTE]", ">"] + ["> " + x for x in block], "", repo, "v2"))
    assert "<figure" not in nested and f"    *Costs \\$5 ([figure code]({code_url})," in nested
    # without a repo_url nothing is rewritten (the same as for ordinary lines)
    assert "](../tools/figures/fig_x.py#L10)" in "\n".join(hook._convert(block, "", "", "main"))


def test_figure_loaders_stop_at_the_cutoff():
    with pytest.raises(C.FigureDataError):
        C.bars("D1", end="2025-01-02", synthetic=True)
    with pytest.raises(C.FigureDataError):
        C.run_table("protocol/trend_core_holdout", "holdout/stats.csv")
    b = C.bars("D1", synthetic=True)
    assert b["available_at"].max() <= C.CUTOFF_END
    assert b.attrs["figure_source"] == "synthetic"
    assert "synthetic" in C.source_label("D1", synthetic=True)
    with C.import_guard(), pytest.raises(C.FigureDataError):
        C.bars("D1", synthetic=True)
