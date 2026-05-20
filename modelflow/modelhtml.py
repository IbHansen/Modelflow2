# -*- coding: utf-8 -*-
"""
modelhtml.py — HTML report generation for ModelFlow model objects.

Currently contains:
  MakeModelReport  –  self-contained HTML report for Makemodel / Listmodels
"""

from dataclasses import dataclass
from typing import List, Any
import re
import html as _mmr_html


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mmr_slugify(text: str) -> str:
    """URL-safe anchor slug from a heading string."""
    text = re.sub(r'[^\w\s-]', '', text.lower())
    return re.sub(r'[\s_]+', '-', text).strip('-') or 'section'


def _mmr_inline_md(text: str) -> str:
    """Bold, italic, code and link markdown on already-escaped HTML text."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*([^*\n]+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'`([^`]+?)`', r'<code>\1</code>', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    return text


def _mmr_has_estimator_tag(line: str) -> bool:
    """True when a source line contains an estimator/est flag."""
    return bool(re.search(r'<[^>]*\b(?:estimator|est)\b[^>]*>', line, flags=re.IGNORECASE))


_BLL_IDENT = re.compile(r'\b([A-Za-z][A-Za-z0-9_]*)\b')


def _mmr_annotate_bll(text: str, var_desc: dict) -> str:
    """HTML-escape BLL text, wrapping known variable names in tooltip <abbr>."""
    if not var_desc:
        return _mmr_html.escape(text)
    esc = _mmr_html.escape
    parts = []
    last = 0
    for m in _BLL_IDENT.finditer(text):
        name = m.group(1)
        parts.append(esc(text[last:m.start()]))
        desc = var_desc.get(name.upper(), '') or var_desc.get(name, '')
        if desc:
            parts.append(
                f'<abbr class="mmr-var" title="{esc(desc)}">{esc(name)}</abbr>'
            )
        else:
            parts.append(esc(name))
        last = m.end()
    parts.append(esc(text[last:]))
    return ''.join(parts)


def _mmr_est_panel(rec: dict, idx: int, plot_format: str) -> str:
    """Return a collapsible HTML panel for one estimation record."""
    esc = _mmr_html.escape
    est = rec.get('estimator_object')
    frml = rec.get('frmlname', '')
    var = getattr(est, 'endo_var', frml) if est else frml
    desc = ''
    if est is not None:
        omdl = getattr(est, 'omodel', None)
        if omdl is not None:
            desc = getattr(omdl, 'var_description', {}).get(var, '')
    caption = f"Estimation: {var}" + (f"  —  {desc}" if desc else "")

    if est is not None and hasattr(est, 'mfresult'):
        try:
            body = est.mfresult.get_html_report(plot_format=plot_format)
        except Exception as exc:
            body = f'<p class="mmr-error">Report error: {esc(str(exc))}</p>'
    elif est is not None and hasattr(est, 'get_html_report'):
        try:
            body = est.get_html_report(plot_format=plot_format)
        except Exception as exc:
            body = f'<p class="mmr-error">Report error: {esc(str(exc))}</p>'
    else:
        org = rec.get('original_expression', '')
        body = f'<p><em>Identity equation</em></p><pre><code>{esc(org)}</code></pre>'

    return (
        f'<button class="mmr-accordion" id="est-{idx}">{esc(caption)}</button>'
        f'<div class="mmr-panel">'
        f'{body}'
        f'<br><a class="mmr-top" href="#top">↑ Back to top</a>'
        f'</div>'
    )


def _mmr_md_to_html(
    md_text: str,
    est_records: list,
    plot_format: str = "svg",
    report_all: bool = False,
    var_desc: dict = None,
) -> tuple:
    """Convert markdown to HTML, inserting collapsible estimation panels.

    Returns ``(html: str, toc: list[(anchor, heading_text)])``.
    """
    esc = _mmr_html.escape
    lines = md_text.splitlines()
    out: List[str] = []
    toc: List[tuple] = []
    anchor_counts: dict = {}
    rec_i = 0
    panel_i = 0

    in_p = in_ul = in_ol = in_bll = in_fence = in_table = False
    tbl_hdr_done = False

    def ep():
        nonlocal in_p
        if in_p:
            out.append('</p>'); in_p = False

    def eul():
        nonlocal in_ul
        if in_ul:
            out.append('</ul>'); in_ul = False

    def eol():
        nonlocal in_ol
        if in_ol:
            out.append('</ol>'); in_ol = False

    def ebll():
        nonlocal in_bll
        if in_bll:
            out.append('</code></pre>'); in_bll = False

    def etbl():
        nonlocal in_table, tbl_hdr_done
        if in_table:
            out.append('</tbody></table>')
            in_table = tbl_hdr_done = False

    def end_all():
        ep(); eul(); eol(); ebll(); etbl()

    for raw in lines:
        s = raw.strip()

        # code fence (```)
        if s.startswith('```'):
            if in_fence:
                out.append('</code></pre>'); in_fence = False
            else:
                end_all()
                lang = s[3:].strip()
                cls = f' class="language-{esc(lang)}"' if lang else ''
                out.append(f'<pre><code{cls}>'); in_fence = True
            continue
        if in_fence:
            out.append(esc(raw)); continue

        # empty line
        if not s:
            end_all(); continue

        # headings ## through ####
        hm = re.match(r'^(#{2,4})\s+(.+)$', s)
        if hm:
            end_all()
            level = len(hm.group(1))
            text = hm.group(2).strip()
            if level == 2:
                base = _mmr_slugify(text)
                n = anchor_counts.get(base, 0)
                anchor = base if n == 0 else f'{base}-{n}'
                anchor_counts[base] = n + 1
                toc.append((anchor, text))
                out.append(f'<h2 id="{esc(anchor)}">{_mmr_inline_md(esc(text))}</h2>')
            else:
                out.append(f'<h{level}>{_mmr_inline_md(esc(text))}</h{level}>')
            continue

        # BLL continuation >>
        if s.startswith('>>'):
            content = s[2:].strip()
            if not in_bll:
                end_all()
                out.append('<pre class="mmr-bll"><code>'); in_bll = True
            out.append(_mmr_annotate_bll(content, var_desc))
            continue

        # BLL equation >
        if s.startswith('>'):
            content = s[1:].strip()
            has_est = _mmr_has_estimator_tag(raw)
            if not in_bll:
                end_all()
                out.append('<pre class="mmr-bll"><code>'); in_bll = True
            out.append(_mmr_annotate_bll(content, var_desc))
            if has_est and rec_i < len(est_records):
                ebll()
                out.append(_mmr_est_panel(est_records[rec_i], panel_i, plot_format))
                panel_i += 1; rec_i += 1
            continue

        # markdown table row
        if s.startswith('|'):
            cells = [c.strip() for c in s.strip('|').split('|')]
            is_sep = bool(cells) and all(re.match(r'^[-: ]+$', c) for c in cells if c)
            if is_sep:
                if in_table and not tbl_hdr_done:
                    for j in range(len(out) - 1, -1, -1):
                        if out[j].startswith('<tr>'):
                            out[j] = out[j].replace('<td>', '<th>').replace('</td>', '</th>')
                            break
                    out.append('</thead><tbody>')
                    tbl_hdr_done = True
                continue
            if not in_table:
                end_all()
                out.append('<table class="mmr-table"><thead>')
                in_table = True; tbl_hdr_done = False
            row = '<tr>' + ''.join(
                f'<td>{_mmr_inline_md(esc(c))}</td>' for c in cells
            ) + '</tr>'
            out.append(row)
            continue

        if in_table:
            etbl()

        # unordered list
        m = re.match(r'^[-*+]\s+(.+)$', s)
        if m:
            ep(); eol()
            if not in_ul:
                out.append('<ul>'); in_ul = True
            out.append(f'<li>{_mmr_inline_md(esc(m.group(1)))}</li>')
            continue

        # ordered list
        m = re.match(r'^\d+[.)]\s+(.+)$', s)
        if m:
            ep(); eul()
            if not in_ol:
                out.append('<ol>'); in_ol = True
            out.append(f'<li>{_mmr_inline_md(esc(m.group(1)))}</li>')
            continue

        # paragraph
        eul(); eol()
        if not in_p:
            out.append('<p>'); in_p = True
        else:
            out.append(' ')
        out.append(_mmr_inline_md(esc(s)))

    if in_fence:
        out.append('</code></pre>')
    end_all()

    # leftover estimation records not matched to any source line
    while rec_i < len(est_records):
        rec = est_records[rec_i]
        if rec.get('estimator_object') is not None or report_all:
            out.append(_mmr_est_panel(rec, panel_i, plot_format))
            panel_i += 1
        rec_i += 1

    return '\n'.join(out), toc


# ---------------------------------------------------------------------------
# CSS / JS templates
# ---------------------------------------------------------------------------

_MMREPORT_CSS = """\
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, Roboto, sans-serif;
  font-size: 1rem; line-height: 1.65; color: #1a1a1a; background: #fff;
  max-width: 980px; margin: 0 auto; padding: 2.5rem 2rem 5rem;
}
a { color: #2563eb; text-decoration: none; }
a:hover { text-decoration: underline; }
h1 {
  font-size: 2rem; font-weight: 700; letter-spacing: -.02em;
  border-bottom: 2px solid #1a1a1a; padding-bottom: .5rem; margin-bottom: 1.5rem;
}
h2 {
  font-size: 1.35rem; font-weight: 600; margin: 2.5rem 0 .75rem;
  border-bottom: 1px solid #e0e0e0; padding-bottom: .3rem; scroll-margin-top: 1rem;
}
h3 { font-size: 1.1rem; font-weight: 600; margin: 1.5rem 0 .5rem; }
h4 { font-size: 1rem; font-weight: 600; margin: 1rem 0 .35rem; }
p { margin: .65rem 0; }
strong { font-weight: 600; }
code {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  font-size: .875em; background: #f5f5f5; padding: .1em .3em; border-radius: 3px;
}
pre {
  background: #f5f5f5; border: 1px solid #e0e0e0; border-radius: 5px;
  padding: 1rem; overflow-x: auto; margin: 1rem 0; line-height: 1.45;
}
pre code { background: none; padding: 0; font-size: .875rem; }
pre.mmr-bll { background: #f0f5fb; border-left: 3px solid #2563eb; border-radius: 0 5px 5px 0; }
abbr.mmr-var {
  text-decoration: none; border-bottom: 1px dotted #2563eb;
  cursor: help; position: relative;
}
abbr.mmr-var:hover::after {
  content: attr(title);
  position: absolute; left: 0; top: 1.4em;
  background: #1a1a1a; color: #fff;
  padding: .2rem .5rem; border-radius: 4px;
  font-size: .78rem; white-space: nowrap; z-index: 20;
  pointer-events: none;
}
ul, ol { padding-left: 1.5rem; margin: .65rem 0; }
li { margin: .25rem 0; }
.mmr-toolbar { display: flex; gap: .5rem; flex-wrap: wrap; margin: 1.25rem 0 1.5rem; }
.mmr-btn {
  background: #1a1a1a; color: #fff; border: none;
  padding: .45rem 1rem; border-radius: 4px; cursor: pointer; font-size: .875rem;
}
.mmr-btn:hover { background: #333; }
.mmr-btn.sec { background: #fff; color: #1a1a1a; border: 1px solid #ccc; }
.mmr-btn.sec:hover { background: #f5f5f5; }
.mmr-toc {
  background: #fafafa; border: 1px solid #e4e4e7;
  border-radius: 6px; padding: 1.25rem 1.5rem; margin: 1rem 0 2rem;
}
.mmr-toc-title {
  font-size: .8rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .08em; color: #666; margin-bottom: .75rem;
}
.mmr-toc ul { list-style: none; padding: 0; margin: 0; }
.mmr-toc li { margin: .3rem 0; font-size: .9rem; }
table.mmr-table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .9rem; }
table.mmr-table th,
table.mmr-table td { border: 1px solid #e0e0e0; padding: .45rem .75rem; text-align: left; vertical-align: top; }
table.mmr-table th { background: #f5f5f5; font-weight: 600; }
table.mmr-table tr:nth-child(even) { background: #fafafa; }
.mmr-accordion {
  background: #f5f5f5; border: 1px solid #e0e0e0; color: #1a1a1a;
  cursor: pointer; padding: .875rem 1rem; width: 100%; border-radius: 5px;
  text-align: left; font-size: .95rem; font-weight: 500;
  margin: 1rem 0 0; transition: background .15s;
}
.mmr-accordion::before { content: "\25b6  "; font-size: .75em; opacity: .5; }
.mmr-accordion.active::before { content: "\25bc  "; }
.mmr-accordion:hover { background: #ebebeb; }
.mmr-accordion.active { border-radius: 5px 5px 0 0; }
.mmr-panel {
  display: none; overflow: hidden; padding: 1rem 1.25rem;
  border: 1px solid #e0e0e0; border-top: none;
  border-radius: 0 0 5px 5px; background: #fff;
}
.mmr-top { display: inline-block; margin-top: 1rem; font-size: .8rem; color: #666; }
.mmr-error { color: #dc2626; font-style: italic; }
@media print {
  .mmr-toolbar, .mmr-top { display: none !important; }
  .mmr-accordion {
    cursor: default; background: transparent !important; border: none !important;
    font-weight: bold; padding: 0;
  }
  .mmr-accordion::before { content: "" !important; }
  .mmr-panel { display: block !important; border: none !important; padding: .5rem 0; }
  .mmr-toc { page-break-after: always; }
  pre { white-space: pre-wrap; word-break: break-all; }
}"""

_MMREPORT_JS = """\
function mmrToggleToc() {
  var el = document.getElementById("mmr-toc-body");
  el.style.display = (el.style.display === "none") ? "" : "none";
}
function mmrExpandAll() {
  document.querySelectorAll(".mmr-accordion").forEach(function(a) {
    a.classList.add("active");
    a.nextElementSibling.style.display = "block";
  });
}
function mmrCollapseAll() {
  document.querySelectorAll(".mmr-accordion").forEach(function(a) {
    a.classList.remove("active");
    a.nextElementSibling.style.display = "none";
  });
}
function mmrPrint() { mmrExpandAll(); setTimeout(function() { window.print(); }, 300); }
function mmrDownload(fn) {
  var blob = new Blob([document.documentElement.outerHTML], {type: "text/html"});
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  a.href = url; a.download = fn; a.click();
  URL.revokeObjectURL(url);
}
document.addEventListener("DOMContentLoaded", function() {
  document.querySelectorAll(".mmr-accordion").forEach(function(btn) {
    btn.addEventListener("click", function() {
      this.classList.toggle("active");
      var p = this.nextElementSibling;
      p.style.display = (p.style.display === "block") ? "none" : "block";
    });
  });
  function mmrOpenForHash(hash) {
    if (!hash || hash === "#top") return;
    var el = document.querySelector(hash);
    if (el && el.classList.contains("mmr-accordion")) {
      el.classList.add("active");
      var p = el.nextElementSibling;
      if (p) p.style.display = "block";
      setTimeout(function() { el.scrollIntoView({behavior: "smooth", block: "start"}); }, 60);
    }
  }
  mmrOpenForHash(window.location.hash);
  window.addEventListener("hashchange", function() { mmrOpenForHash(window.location.hash); });
});"""


def _mmreport_build_html(title: str, toc: list, body: str, filename: str) -> str:
    """Assemble the complete, self-contained HTML document."""
    esc = _mmr_html.escape
    toc_items = '\n'.join(
        f'      <li><a href="#{esc(a)}">{esc(t)}</a></li>'
        for a, t in toc
    )
    return (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<title>{esc(title)}</title>\n'
        '<style>\n'
        + _MMREPORT_CSS
        + '\n</style>\n'
        '<script>\n'
        + _MMREPORT_JS
        + '\n</script>\n'
        '</head>\n'
        f'<body id="top">\n'
        f'<h1>{esc(title)}</h1>\n'
        '<div class="mmr-toolbar">\n'
        '  <button class="mmr-btn" onclick="mmrToggleToc()">Contents</button>\n'
        '  <button class="mmr-btn" onclick="mmrExpandAll()">Expand all</button>\n'
        '  <button class="mmr-btn" onclick="mmrCollapseAll()">Collapse all</button>\n'
        '  <button class="mmr-btn sec" onclick="mmrPrint()">Print</button>\n'
        f'  <button class="mmr-btn sec" onclick="mmrDownload(\'{esc(filename)}\')">'
        'Download</button>\n'
        '</div>\n'
        '<nav class="mmr-toc">\n'
        '  <div class="mmr-toc-title">Contents</div>\n'
        '  <div id="mmr-toc-body">\n'
        '    <ul>\n'
        + toc_items
        + '\n    </ul>\n'
        '  </div>\n'
        '</nav>\n'
        '<main>\n'
        + body
        + '\n</main>\n'
        '</body>\n'
        '</html>'
    )


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

@dataclass
class MakeModelReport:
    """Self-contained HTML report for a Makemodel or Listmodels instance.

    Renders ``original_statements`` markdown as HTML, builds a collapsible
    Table of Contents from ``##`` headings, and inserts a collapsible
    estimation panel immediately after each estimated equation line.

    Parameters
    ----------
    makemodel_obj : Makemodel or Listmodels
        Source model(s) to document.
    title : str, optional
        Report title. Defaults to ``modelname`` or ``"Model Report"``.
    plot_format : {"svg", "png"}, default ``"svg"``
        Embedded plot format.
    report_all : bool, default ``False``
        Include identity equations as minimal panels alongside estimated ones.

    Examples
    --------
    >>> from modelhtml import MakeModelReport
    >>> rpt = MakeModelReport(mm, title="Consumption model")
    >>> rpt.save()                    # writes html/consumption-model_report.html
    >>> rpt.save(open_file=True)      # same, opens in browser
    >>> rpt.show()                    # open a temp copy in the browser
    """

    makemodel_obj: Any
    title: str = ""
    plot_format: str = "svg"
    report_all: bool = False

    def _effective_title(self) -> str:
        if self.title:
            return self.title
        return getattr(self.makemodel_obj, 'modelname', '') or "Model Report"

    def _iter_sources(self):
        obj = self.makemodel_obj
        if hasattr(obj, 'makemodels'):   # Listmodels
            yield from obj.makemodels
        else:                            # Makemodel or compatible duck type
            yield obj

    def _build_html(self) -> str:
        title = self._effective_title()
        filename = f'{_mmr_slugify(title)}_report.html'
        all_toc: List[tuple] = []
        all_parts: List[str] = []
        is_multi = hasattr(self.makemodel_obj, 'makemodels')

        for mex in self._iter_sources():
            sub_name = getattr(mex, 'modelname', '') or ''
            md_text = getattr(mex, 'original_statements', '') or ''
            est_recs = list(getattr(mex, 'estimation_records', []))

            if is_multi and sub_name:
                anchor = _mmr_slugify(sub_name)
                all_toc.append((anchor, sub_name))
                all_parts.append(
                    f'<h2 id="{_mmr_html.escape(anchor)}">'
                    f'{_mmr_html.escape(sub_name)}</h2>'
                )

            var_desc = getattr(mex, 'var_description', None) or {}
            body_html, toc_entries = _mmr_md_to_html(
                md_text, est_recs, self.plot_format, self.report_all, var_desc
            )
            all_toc.extend(toc_entries)
            all_parts.append(body_html)

        return _mmreport_build_html(
            title=title,
            toc=all_toc,
            body='\n'.join(all_parts),
            filename=filename,
        )

    def save(
        self,
        path: str = "html",
        filename: str = "",
        open_file: bool = False,
    ) -> str:
        """Write the HTML report to *path/filename*.

        Parameters
        ----------
        path : str, default ``"html"``
            Output directory (created if needed).
        filename : str, optional
            File name inside *path*. Derived from the title when omitted.
        open_file : bool, default ``False``
            Open the saved file in the default browser after writing.

        Returns
        -------
        str
            Absolute path of the saved file.
        """
        from pathlib import Path
        import webbrowser
        html_text = self._build_html()
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not filename:
            filename = f'{_mmr_slugify(self._effective_title())}_report.html'
        full = out_dir / filename
        full.write_text(html_text, encoding='utf-8')
        print(f"✔ Report saved → {full.resolve()}")
        if open_file:
            webbrowser.open(f"file://{full.resolve()}")
        return str(full)

    def show(self) -> None:
        """Write to a temp file and open in the default browser."""
        import tempfile
        import webbrowser
        from pathlib import Path
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.html', encoding='utf-8', delete=False
        ) as fh:
            fh.write(self._build_html())
        webbrowser.open(f"file://{Path(fh.name).resolve()}")

    def _repr_html_(self) -> str:
        """Jupyter rich-display: embeds the full HTML page."""
        return self._build_html()
