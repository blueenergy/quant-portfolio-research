"""HTML report shell and numeric formatting primitives for analysis reports."""

from __future__ import annotations

from html import escape
from typing import Any, Optional

__all__ = ["as_float", "fmt_pct", "fmt_num", "html_document"]


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_pct(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "-"
    return f"{number * 100:+.2f}%"


def fmt_num(value: Any, digits: int = 2) -> str:
    number = as_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


_CSS = """
body { margin: 0; padding: 32px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; color: #172033; background: #f6f8fb; line-height: 1.55; }
main { max-width: 1440px; margin: 0 auto; padding: 32px; background: #fff; border: 1px solid #e6eaf0; border-radius: 14px; }
.hero { border-bottom: 1px solid #edf0f5; margin-bottom: 22px; padding-bottom: 18px; }
.eyebrow { margin: 0 0 4px; color: #64748b; font-size: 13px; letter-spacing: .04em; text-transform: uppercase; }
h1 { margin: 0; font-size: 30px; }
h2 { margin: 34px 0 14px; font-size: 21px; }
.subtitle { margin: 8px 0 0; color: #64748b; }
.meta { color: #64748b; font-size: 13px; margin: 4px 0; }
.summary-card { border: 1px solid #e6eaf0; border-radius: 12px; padding: 18px; background: #fbfcfe; margin: 18px 0; display: inline-block; }
.warning { border: 1px solid #f0d9a8; background: #fff8e6; border-radius: 12px; padding: 14px 18px; margin: 18px 0; color: #6b4e16; }
.warning h2 { margin: 0 0 6px; font-size: 18px; }
.warning p { margin: 0; }
.label { color: #475569; font-weight: 700; }
.big { font-size: 28px; font-weight: 800; margin: 4px 0; }
.table-wrap { overflow-x: auto; border: 1px solid #e6eaf0; border-radius: 10px; margin-bottom: 18px; max-height: 840px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { padding: 10px 12px; border-bottom: 1px solid #edf0f5; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { background: #f8fafc; color: #475569; font-weight: 700; position: sticky; top: 0; cursor: pointer; user-select: none; }
th:hover { background: #eef2f7; }
tbody tr:nth-child(even) td { background: #fbfcfe; }
tbody tr:hover td { background: #eef6ff; }
.strategy { font-weight: 700; color: #0f3f75; }
td.pos { color: #067a4e; font-weight: 600; }
td.neg { color: #c0392b; font-weight: 600; }
code { padding: 2px 6px; border-radius: 5px; background: #eef2f7; color: #0f3f75; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .92em; }
li { margin: 6px 0; }
.foot { color: #9aa3af; font-size: 12px; margin-top: 24px; }
"""

_SORT_JS = """
function sortTable(id,col){
 var t=document.getElementById(id),tb=t.tBodies[0],rows=Array.prototype.slice.call(tb.rows);
 var asc=t.getAttribute('data-sortcol')==col?!(t.getAttribute('data-asc')=='1'):true;
 rows.sort(function(a,b){
  var x=parseFloat(a.cells[col].getAttribute('data-sort')),y=parseFloat(b.cells[col].getAttribute('data-sort'));
  if(isNaN(x)&&isNaN(y)){x=a.cells[col].innerText;y=b.cells[col].innerText;return asc?(x<y?-1:x>y?1:0):(x>y?-1:x<y?1:0);}
  return asc?x-y:y-x;});
 rows.forEach(function(r){tb.appendChild(r);});
 t.setAttribute('data-sortcol',col);t.setAttribute('data-asc',asc?'1':'0');
}
"""


def html_document(title: str, body: str) -> str:
    """Wrap ``body`` in the shared HTML shell (CSS + sort script)."""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>{_CSS}</style>
</head>
<body>
  <main>{body}</main>
  <script>{_SORT_JS}</script>
</body>
</html>
"""
