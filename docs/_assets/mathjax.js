// MathJax settings for the MkDocs site. pymdownx.arithmatex (generic mode) turns ```math
// fences into <div class="arithmatex">\[...\]</div> and inline $...$ into
// <span class="arithmatex">\(...\)</span>; only those elements are typeset.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Re-typeset after Material's instant navigation swaps the page content.
if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    if (window.MathJax && MathJax.startup) {
      MathJax.startup.output.clearCache();
      MathJax.typesetClear();
      MathJax.texReset();
      MathJax.typesetPromise();
    }
  });
}
