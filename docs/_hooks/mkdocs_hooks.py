"""MkDocs build hooks for the Aurum documentation (see ``hooks:`` in ``mkdocs.yml``).

The Markdown in ``docs/`` is written for GitHub first. This hook adapts five GitHub
conventions when the pages are built into a site, without changing the source files:

* Relative links that leave ``docs/`` (``../SPEC.md``, ``../aurum/cli.py``, ...) point at
  files that are not part of the site. They are rewritten to URLs on the GitHub repository
  (``repo_url`` + ``extra.source_ref``), so they work in both places.
* GitHub alert blocks (``> [!WARNING]`` followed by ``> ...`` lines) become admonitions.
* ``<details><summary>...</summary> ... </details>`` blocks become collapsible
  ``pymdownx.details`` blocks, so the Markdown inside them is rendered.
* Theme-aware figures. GitHub picks the image of a ``<picture>`` block with
  ``prefers-color-scheme``; Material's light/dark toggle does not change that media query,
  so the canonical embed (see tools/figures/README.md)::

      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="_assets/figures/p/x-dark.png">
        <img src="_assets/figures/p/x-light.png" alt="Alt text">
      </picture>

      *Caption.*

  becomes a ``<figure>`` holding ``![Alt text](..x-light.png#only-light)`` and
  ``![Alt text](..x-dark.png#only-dark)`` (Material shows exactly one of them for the active
  palette) with the italic caption as its ``<figcaption>``. Paths stay relative to the page.
  A ``<picture>`` block that does not match the format is left alone and logged as a
  warning (``mkdocs build --strict`` fails on it).
* Currency dollars. ``pymdownx.arithmatex`` treats any ``$...$`` pair on a line as inline
  math, so prose such as ``+$22.9k gross on $100k ... -$0.7k`` would be typeset. A ``$`` is
  kept as a math delimiter only when it pairs on the same line under pandoc's rule (opening
  ``$`` followed by a non-space, closing ``$`` preceded by a non-space and not followed by a
  digit); every other ``$`` is escaped (``\\$``, which arithmatex registers as an escape).
  Lines containing ``$$``, display blocks between ``$$`` lines and lines starting with an
  HTML tag (raw HTML is neither escaped nor typeset) are left untouched. Inside alerts,
  ``<details>`` and indented blocks a figure is emitted as plain image and caption
  paragraphs, because md_in_html only applies to unindented HTML.

Fenced code blocks and inline code spans are left untouched.
"""
from __future__ import annotations

import html
import logging
import posixpath
import re

log = logging.getLogger("mkdocs.hooks.aurum")

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

_PICTURE_OPEN = re.compile(r"^(\s*)<picture\b[^>]*>\s*$", re.I)
_PICTURE_SOURCE = re.compile(r"<source\b[^>]*>", re.I)
_PICTURE_IMG = re.compile(r"<img\b[^>]*>", re.I)
_CAPTION = re.compile(r"^\s*\*(?!\*)(\S(?:.*\S)?)\*\s*$|^\s*_(?!_)(\S(?:.*\S)?)_\s*$")

# spans in which a "$" is never a math delimiter: code spans, HTML tags, link targets
_NO_MATH = re.compile(r"(`+)(?:(?!\1).)+?\1|<[^>\n]*>|\]\([^)\n]*\)")


def _outside_link(match: re.Match, page_dir: str, repo: str, ref: str) -> str:
    target = posixpath.normpath(posixpath.join(page_dir, match.group(2)))
    if not target.startswith("../"):
        return match.group(0)  # still inside docs/: leave it to MkDocs
    path = target[3:]
    kind = "tree" if match.group(2).endswith("/") else "blob"
    return f"{match.group(1)}{repo}/{kind}/{ref}/{path}{match.group(3) or ''}{match.group(4)}"


# ---------------------------------------------------------------------------------------
# currency dollars
# ---------------------------------------------------------------------------------------
def _escape_stray_dollars(text: str) -> str:
    """Escape every ``$`` in ``text`` (no code/tags) that is not part of a pandoc-style pair."""
    pos = [i for i, ch in enumerate(text) if ch == "$" and (i == 0 or text[i - 1] != "\\")]
    if not pos:
        return text
    stray: list[int] = []
    i = 0
    while i < len(pos):
        o = pos[i]
        if i + 1 < len(pos):
            c = pos[i + 1]
            after = text[c + 1] if c + 1 < len(text) else ""
            if (c > o + 1 and not text[o + 1].isspace() and not text[c - 1].isspace()
                    and not after.isdigit()):
                i += 2
                continue
        stray.append(o)
        i += 1
    for k in reversed(stray):
        text = text[:k] + "\\" + text[k:]
    return text


def _protect_dollars(line: str) -> str:
    # Raw HTML lines are neither Markdown-escaped nor typeset by arithmatex: leave them be.
    if "$" not in line or "$$" in line or line.lstrip().startswith("<"):
        return line
    out, last = [], 0
    for m in _NO_MATH.finditer(line):
        out.append(_escape_stray_dollars(line[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(_escape_stray_dollars(line[last:]))
    return "".join(out)


# ---------------------------------------------------------------------------------------
# theme-aware figures
# ---------------------------------------------------------------------------------------
def _attr(tag: str, name: str) -> str | None:
    m = re.search(rf"\b{name}\s*=\s*\"([^\"]*)\"", tag, re.I)
    return html.unescape(m.group(1)).strip() if m else None


def _md_alt(text: str) -> str:
    return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("\n", " ")


def _picture(lines: list[str], i: int, page: str, nested: bool) -> tuple[list[str], int] | None:
    """Convert the ``<picture>`` block starting at ``lines[i]``; return (new lines, next i).

    At the top level the result is an unindented ``<figure markdown="span">`` (md_in_html).
    Inside alerts, ``<details>`` or any indented context md_in_html does not apply, so the two
    images and the italic caption are emitted as plain Markdown paragraphs instead.
    """
    indent = _PICTURE_OPEN.match(lines[i]).group(1)
    j = i
    while j < len(lines) and "</picture>" not in lines[j].lower():
        j += 1
    if j >= len(lines):
        log.warning("%s:%d: <picture> without </picture>; left as raw HTML", page, i + 1)
        return None
    block = "\n".join(lines[i:j + 1])
    dark = None
    for tag in _PICTURE_SOURCE.findall(block):
        if (_attr(tag, "media") or "").replace(" ", "") == "(prefers-color-scheme:dark)":
            dark = _attr(tag, "srcset")
    imgs = _PICTURE_IMG.findall(block)
    light = _attr(imgs[0], "src") if imgs else None
    alt = (_attr(imgs[0], "alt") or "") if imgs else ""
    if not (dark and light and alt):
        log.warning("%s:%d: <picture> needs a (prefers-color-scheme: dark) <source srcset>, an <img src> "
                    "and alt text (see tools/figures/README.md); left as raw HTML", page, i + 1)
        return None
    caption = None
    k = j + 1
    if k + 1 < len(lines) and lines[k].strip() == "":
        cm = _CAPTION.match(lines[k + 1])
        if cm:
            caption = cm.group(1) or cm.group(2)
            k += 2
    a = _md_alt(alt)
    images = [f"![{a}]({light}#only-light){{ loading=lazy }}", f"![{a}]({dark}#only-dark){{ loading=lazy }}"]
    cap = f"*{_protect_dollars(caption)}*" if caption else None
    if nested or indent:
        out = [indent + x for x in images] + ([indent.rstrip(), indent + cap] if cap else []) + [""]
    else:
        out = ['<figure class="aurum-figure" markdown="span">', *images]
        if cap:
            out.append(f'<figcaption markdown="span">{cap}</figcaption>')
        out += ["</figure>", ""]
    return out, k


# ---------------------------------------------------------------------------------------
def _convert(lines: list[str], page_dir: str, repo: str, ref: str, page: str = "",
             nested: bool = False) -> list[str]:
    out: list[str] = []
    i, in_fence, fence, in_display = 0, False, "", False
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
        if line.strip() == "$$":
            in_display = not in_display
            out.append(line)
            i += 1
            continue
        if in_display:
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
            out.extend(("    " + b) if b.strip() else "" for b in _convert(body, page_dir, repo, ref, page, True))
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
            out.extend(("    " + b) if b.strip() else "" for b in _convert(body, page_dir, repo, ref, page, True))
            out.append("")
            i = j + 1
            continue
        if _PICTURE_OPEN.match(line):
            converted = _picture(lines, i, page, nested)
            if converted is not None:
                new, i = converted
                out.extend(new)
                continue
        line = _protect_dollars(line)
        if repo:
            line = _LINK.sub(lambda mm: _outside_link(mm, page_dir, repo, ref), line)
        out.append(line)
        i += 1
    return out


def on_page_markdown(markdown: str, page, config, files) -> str:
    repo = (config.get("repo_url") or "").rstrip("/")
    ref = (config.get("extra") or {}).get("source_ref", "main")
    page_dir = posixpath.dirname(page.file.src_uri)
    return "\n".join(_convert(markdown.split("\n"), page_dir, repo, ref, page.file.src_uri))
