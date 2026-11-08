"""Build an offline HTML reading edition from a Gym primer Markdown source.

    python3 tools/build_reading_edition.py modules/t3-sim-to-rl/s06-primer.md \
        --eyebrow "T3 · S06 · 通读讲义" --subtitle "先通读全部知识点，再进入练习单"

Requires pandoc on PATH. Inputs, all optional except the Markdown:
  reading-assets/<stem>/diagrams/<n>.svg   n-th ```mermaid block -> inline SVG (count must match)
  reading-assets/<stem>/demos/<name>.html  ```reader-demo <name>``` block -> widget fragment
  reading-assets/<stem>/script.js          appended once at the end of <body>
  reading-assets/<stem>/extra.css          appended after the shared tools/reading-edition.css
The first "# " heading becomes the page title. Edit the Markdown, never the generated HTML.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
FOOTER = ("Markdown 是编辑源稿；本文件由 tools/build_reading_edition.py 生成。公式使用原生 MathML，图为内嵌 SVG，"
          "不依赖网络脚本。外链材料需联网，仓库内链接需保留相对目录。")


def build(src: Path, eyebrow: str, subtitle: str) -> Path:
    md = src.read_text()
    title = re.search(r"^# (.+)$", md, re.M).group(1)
    body_md = re.sub(r"^# .+\n", "", md, count=1, flags=re.M)
    template = '<nav aria-label="讲义目录"><details open><summary>阅读目录</summary>$toc$</details></nav><main>$body$</main>'
    tpl = HERE / ".reading-edition-template.html"
    tpl.write_text(template)
    try:
        body = subprocess.run(
            ["pandoc", "-f", "gfm+tex_math_dollars", "-t", "html5", "--mathml", "--toc", "--toc-depth=3",
             "--template", str(tpl), "--wrap=none"],
            input=body_md, capture_output=True, text=True, check=True).stdout
    finally:
        tpl.unlink()
    body = re.sub(r"(<table>.*?</table>)", r'<div class="table-wrap">\1</div>', body, flags=re.S)

    assets = src.parent / "reading-assets" / src.stem
    mermaid = re.findall(r'<pre class="mermaid"><code>.*?</code></pre>', body, flags=re.S)
    svgs = sorted((assets / "diagrams").glob("*.svg"), key=lambda p: int(p.stem)) if (assets / "diagrams").exists() else []
    if len(mermaid) != len(svgs):
        raise SystemExit(f"{len(mermaid)} mermaid blocks but {len(svgs)} SVGs in {assets / 'diagrams'}")
    for block, svg in zip(mermaid, svgs):
        body = body.replace(block, f'<figure class="diagram">{svg.read_text().strip()}</figure>', 1)
    for name in re.findall(r'<pre class="reader-demo"><code>([\w-]+)\n?</code></pre>', body):
        frag = (assets / "demos" / f"{name}.html").read_text().strip()
        body = re.sub(rf'<pre class="reader-demo"><code>{name}\n?</code></pre>', lambda _: frag, body, count=1)

    css = (HERE / "reading-edition.css").read_text().strip()
    extra = (assets / "extra.css").read_text().strip() if (assets / "extra.css").exists() else ""
    script = (assets / "script.js").read_text().strip() if (assets / "script.js").exists() else ""
    sha = hashlib.sha256(md.encode()).hexdigest()
    pandoc = subprocess.run(["pandoc", "--version"], capture_output=True, text=True).stdout.split("\n")[0]
    page = (
        f"<!doctype html>\n<!-- Generated reading edition. Source: {src.name}; SHA256: {sha}. {pandoc}. "
        f"Edit the Markdown source first. -->"
        f'<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{css}{extra}</style></head><body>"
        f'<header><div class="eyebrow">{html.escape(eyebrow)}</div><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></header>'
        f'<div class="layout">{body.replace("</main>", f"<footer>{FOOTER}</footer></main>")}</div>'
        + (f"<script>{script}</script>" if script else "")
        + "</body></html>\n"
    )
    out = src.with_suffix(".html")
    out.write_text(page)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("markdown", type=Path)
    p.add_argument("--eyebrow", required=True)
    p.add_argument("--subtitle", required=True)
    a = p.parse_args()
    print(build(a.markdown, a.eyebrow, a.subtitle))


if __name__ == "__main__":
    main()
