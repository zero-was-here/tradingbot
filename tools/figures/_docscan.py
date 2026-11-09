"""Scan the Markdown docs for images, links, anchors and figure embeds.

Shared by ``tools/figures/build.py --check`` and ``tests/test_docs_assets.py``. Pure text
processing (no MkDocs, no network): fenced code blocks and inline code spans are ignored,
heading anchors follow GitHub's slug rule (identical to the site's
``pymdownx.slugs.slugify(case="lower")`` on every heading in docs/).
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
FIGURES_DIR = DOCS / "_assets" / "figures"

_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_CODE_SPAN = re.compile(r"(`+)(?:(?!\1).)+?\1")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_MD_IMAGE = re.compile(r"!\[(?P<alt>(?:\\.|[^\]\\])*)\]\(\s*<?(?P<src>[^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_MD_LINK = re.compile(r"(?<!!)\[(?:\\.|[^\]\\])*\]\(\s*<?(?P<href>[^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_REF_DEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(?P<href>\S+?)>?(?:\s+.*)?$")
_HTML_ATTR = re.compile(r"<(?P<tag>img|source|a)\b[^>]*?\b(?P<attr>src|srcset|href)\s*=\s*\"(?P<val>[^\"]*)\"", re.I)
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_EXPLICIT_ID = re.compile(r"""<a\s+(?:[^>]*?\s)?(?:id|name)\s*=\s*["']([^"']+)["']|\{\s*#([\w\-:.]+)[^}]*\}""")


def doc_files(include_readme: bool = True) -> list[Path]:
    """``docs/*.md`` (top level, like the site) plus ``README.md``."""
    files = sorted(DOCS.glob("*.md"))
    if include_readme and (REPO / "README.md").exists():
        files.append(REPO / "README.md")
    return files


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def prose_lines(text: str, *, blank_code: bool = True) -> Iterator[tuple[int, str]]:
    """``(line_no, line)`` outside fenced code, with HTML comments and (``blank_code``)
    inline code spans blanked, so their contents never count as links or images."""
    text = _HTML_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    in_fence, fence = False, ""
    for no, line in enumerate(text.split("\n"), 1):
        m = _FENCE.match(line)
        if m:
            ticks = m.group(1)
            if not in_fence:
                in_fence, fence = True, ticks
            elif ticks[0] == fence[0] and len(ticks) >= len(fence) and line.strip().strip(fence[0]) == "":
                in_fence = False
            continue
        if in_fence:
            continue
        yield no, (_CODE_SPAN.sub(lambda mm: " " * len(mm.group(0)), line) if blank_code else line)


def _is_external(target: str) -> bool:
    t = target.strip()
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", t)) or t.startswith("//")


@dataclass
class Ref:
    file: Path
    line: int
    target: str          # as written (may carry #fragment)
    kind: str            # "image" | "link"

    @property
    def path_part(self) -> str:
        return html.unescape(self.target.split("#", 1)[0])

    @property
    def fragment(self) -> str:
        return self.target.split("#", 1)[1] if "#" in self.target else ""

    def resolve(self) -> Path | None:
        p = self.path_part
        if not p:
            return None
        return (self.file.parent / p).resolve()

    def where(self) -> str:
        return f"{rel(self.file)}:{self.line}"


def refs(path: Path) -> list[Ref]:
    """Every local image and link reference in one Markdown file."""
    out: list[Ref] = []
    text = path.read_text(encoding="utf-8")
    for no, line in prose_lines(text):
        for m in _MD_IMAGE.finditer(line):
            out.append(Ref(path, no, m.group("src"), "image"))
        stripped = _MD_IMAGE.sub(" ", line)
        for m in _MD_LINK.finditer(stripped):
            out.append(Ref(path, no, m.group("href"), "link"))
        m = _REF_DEF.match(line)
        if m:
            out.append(Ref(path, no, m.group("href"), "link"))
        for m in _HTML_ATTR.finditer(line):
            kind = "link" if m.group("attr").lower() == "href" else "image"
            val = m.group("val").strip()
            if m.group("attr").lower() == "srcset":
                val = val.split(",")[0].strip().split(" ")[0]
            out.append(Ref(path, no, val, kind))
    return [r for r in out if r.target and not _is_external(r.target)]


# ---------------------------------------------------------------------------------------
# anchors
# ---------------------------------------------------------------------------------------
def _strip_inline(t: str) -> str:
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = t.replace("`", "")
    t = re.sub(r"(\*\*|__)(.+?)\1", r"\2", t)
    t = re.sub(r"(?<!\w)[*_](.+?)[*_](?!\w)", r"\1", t)
    return t


def slug(text: str) -> str:
    """GitHub heading slug (lowercase, punctuation dropped, spaces -> hyphens)."""
    t = _strip_inline(html.unescape(text)).strip().lower()
    t = re.sub(r"\{\s*#[^}]*\}\s*$", "", t).strip()
    t = re.sub(r"[^\w\- ]", "", t)
    return t.replace(" ", "-")


_ANCHOR_CACHE: dict[Path, set[str]] = {}


def anchors(path: Path) -> set[str]:
    """Heading slugs (with GitHub's ``-1``, ``-2`` de-duplication) and explicit ids."""
    path = path.resolve()
    if path in _ANCHOR_CACHE:
        return _ANCHOR_CACHE[path]
    seen: dict[str, int] = {}
    out: set[str] = set()
    for _, line in prose_lines(path.read_text(encoding="utf-8"), blank_code=False):
        for m in _EXPLICIT_ID.finditer(line):
            out.add(m.group(1) or m.group(2))
        h = _HEADING.match(line)
        if not h:
            continue
        s = slug(h.group(2))
        if s in seen:
            seen[s] += 1
            out.add(f"{s}-{seen[s]}")
        else:
            seen[s] = 0
            out.add(s)
    _ANCHOR_CACHE[path] = out
    return out


# ---------------------------------------------------------------------------------------
# <picture> embeds
# ---------------------------------------------------------------------------------------
_PIC_SOURCE = re.compile(r"<source\b[^>]*>", re.I)
_PIC_IMG = re.compile(r"<img\b[^>]*>", re.I)


def _attr(tag: str, name: str) -> str | None:
    m = re.search(rf"\b{name}\s*=\s*\"([^\"]*)\"", tag, re.I)
    return html.unescape(m.group(1)) if m else None


@dataclass
class Picture:
    file: Path
    line: int                      # line of <picture>
    dark: str | None = None        # <source media="(prefers-color-scheme: dark)" srcset=...>
    light: str | None = None       # <img src=...>
    alt: str | None = None
    caption: str | None = None     # the italic line after one blank line
    problems: list[str] = field(default_factory=list)

    def where(self) -> str:
        return f"{rel(self.file)}:{self.line}"


def pictures(path: Path) -> list[Picture]:
    """Parse every ``<picture>`` block and check it against the canonical embed format."""
    lines = path.read_text(encoding="utf-8").split("\n")
    prose = dict(prose_lines("\n".join(lines)))
    out: list[Picture] = []
    i = 0
    while i < len(lines):
        no = i + 1
        if no in prose and lines[i].strip().lower().startswith("<picture"):
            j = i
            while j < len(lines) and "</picture>" not in lines[j].lower():
                j += 1
            block = "\n".join(lines[i:j + 1])
            pic = Picture(path, no)
            if j >= len(lines):
                pic.problems.append("no closing </picture>")
            for tag in _PIC_SOURCE.findall(block):
                media = (_attr(tag, "media") or "").replace(" ", "")
                if media == "(prefers-color-scheme:dark)":
                    pic.dark = _attr(tag, "srcset")
            imgs = _PIC_IMG.findall(block)
            if imgs:
                pic.light = _attr(imgs[0], "src")
                pic.alt = _attr(imgs[0], "alt")
            if not pic.dark:
                pic.problems.append('missing <source media="(prefers-color-scheme: dark)" srcset="...-dark.png">')
            if not pic.light:
                pic.problems.append('missing <img src="...-light.png" alt="...">')
            if not (pic.alt or "").strip():
                pic.problems.append("empty alt text")
            if pic.light and pic.dark and pic.light.replace("-light.", "-dark.") != pic.dark:
                pic.problems.append(f"light/dark pair mismatch ({pic.light} vs {pic.dark})")
            k = j + 1
            if k < len(lines) and lines[k].strip() == "" and k + 1 < len(lines):
                cap = lines[k + 1].strip()
                if re.match(r"^\*[^*\s].*[^*\s]\*$|^\*[^*\s]\*$|^_[^_\s].*[^_\s]_$", cap):
                    pic.caption = cap[1:-1]
            if pic.caption is None:
                pic.problems.append("no italic caption line (*...*) after one blank line")
            out.append(pic)
            i = j + 1
            continue
        i += 1
    return out


__all__ = ["Picture", "Ref", "anchors", "doc_files", "pictures", "prose_lines", "refs", "rel", "slug"]
