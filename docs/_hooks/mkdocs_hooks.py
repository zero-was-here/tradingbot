"""MkDocs build hooks for the Aurum documentation (see ``hooks:`` in ``mkdocs.yml``).

The Markdown in ``docs/`` is written for GitHub first. This hook adapts three GitHub
conventions when the pages are built into a site, without changing the source files:

* Relative links that leave ``docs/`` (``../SPEC.md``, ``../aurum/cli.py``, ...) point at
  files that are not part of the site. They are rewritten to URLs on the GitHub repository
  (``repo_url`` + ``extra.source_ref``), so they work in both places.
* GitHub alert blocks (``> [!WARNING]`` followed by ``> ...`` lines) become admonitions.
* ``<details><summary>...</summary> ... </details>`` blocks become collapsible
  ``pymdownx.details`` blocks, so the Markdown inside them is rendered.

Fenced code blocks are left untouched.
"""
from __future__ import annotations

import posixpath
import re

_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_LINK = re.compile(r"(\]\()(\.\./[^)\s#]*)(#[^)\s]*)?(\))")
_ALERT = re.compile(r"^>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*$")
_ALERT_KIND = {
    "NOTE": ("note", "Note"),
    "TIP": ("tip", "Tip"),
    "IMPORTANT": ("info", "Important"),
    "WARNING": ("warning", "Warning"),
    "CAUTION": ("danger", "Caution"),
}
_SUMMARY = re.compile(r"^\s*<summary>(.*?)</summary>\s*$")


def _outside_link(match: re.Match, page_dir: str, repo: str, ref: str) -> str:
    target = posixpath.normpath(posixpath.join(page_dir, match.group(2)))
    if not target.startswith("../"):
        return match.group(0)  # still inside docs/: leave it to MkDocs
    path = target[3:]
    kind = "tree" if match.group(2).endswith("/") else "blob"
    return f"{match.group(1)}{repo}/{kind}/{ref}/{path}{match.group(3) or ''}{match.group(4)}"


def _convert(lines: list[str], page_dir: str, repo: str, ref: str) -> list[str]:
    out: list[str] = []
    i, in_fence, fence = 0, False, ""
    while i < len(lines):
        line = lines[i]
        m = _FENCE.match(line)
        if m:
            ticks = m.group(1)
            if not in_fence:
                in_fence, fence = True, ticks
            elif ticks[0] == fence[0] and len(ticks) >= len(fence) and line.strip().strip(fence[0]) == "":
                in_fence = False
            out.append(line)
            i += 1
            continue
        if in_fence:
            out.append(line)
            i += 1
            continue
        alert = _ALERT.match(line)
        if alert:
            kind, title = _ALERT_KIND[alert.group(1)]
            body: list[str] = []
            i += 1
            while i < len(lines) and lines[i].startswith(">"):
                body.append(re.sub(r"^> ?", "", lines[i]))
                i += 1
            out.append(f'!!! {kind} "{title}"')
            out.append("")
            out.extend(("    " + b) if b.strip() else "" for b in _convert(body, page_dir, repo, ref))
            out.append("")
            continue
        if line.strip() == "<details>" and i + 1 < len(lines) and _SUMMARY.match(lines[i + 1]):
            title = _SUMMARY.match(lines[i + 1]).group(1).replace('"', "'")
            j = i + 2
            body = []
            while j < len(lines) and lines[j].strip() != "</details>":
                body.append(lines[j])
                j += 1
            out.append(f'??? info "{title}"')
            out.append("")
            out.extend(("    " + b) if b.strip() else "" for b in _convert(body, page_dir, repo, ref))
            out.append("")
            i = j + 1
            continue
        if repo:
            line = _LINK.sub(lambda mm: _outside_link(mm, page_dir, repo, ref), line)
        out.append(line)
        i += 1
    return out


def on_page_markdown(markdown: str, page, config, files) -> str:
    repo = (config.get("repo_url") or "").rstrip("/")
    ref = (config.get("extra") or {}).get("source_ref", "main")
    page_dir = posixpath.dirname(page.file.src_uri)
    return "\n".join(_convert(markdown.split("\n"), page_dir, repo, ref))
