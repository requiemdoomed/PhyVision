"""Render docs/images/photo_guide.png (the framing diagram + framing levels)
from the web photo guide, using a headless browser (requires `playwright`).
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
HTML = """<!doctype html><html data-theme="light"><head><meta charset="utf-8">
<link rel="stylesheet" href="{web}/assets/phyvision.css"><script src="{web}/assets/static_data.js"></script>
<script src="{web}/assets/phyvision.js"></script>
<style>body{{background:#eef1ec;padding:24px;width:1100px}} #wrap{{display:grid;grid-template-columns:340px 1fr;gap:32px;align-items:center}}</style>
</head><body><div id="wrap"><div id="d"></div><div><div class="framing-row" id="f"></div></div></div>
<script>document.getElementById('d').innerHTML=PV.frameDiagram();
var m=PV.static.metrics; document.getElementById('f').innerHTML=PV.framingStrip(m&&m.photo_pipeline_rembg&&m.photo_pipeline_rembg.by_framing);</script>
</body></html>"""


def main():
    tmp = ROOT / "web" / "_guide_render.html"
    tmp.write_text(HTML.format(web=(ROOT / "web").as_uri()))
    out = ROOT / "docs" / "images" / "photo_guide.png"
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1150, "height": 600}, device_scale_factor=2)
            pg.goto(tmp.as_uri())
            pg.wait_for_timeout(800)
            pg.locator("#wrap").screenshot(path=str(out))
            b.close()
    finally:
        tmp.unlink(missing_ok=True)
    print("Wrote", out)


if __name__ == "__main__":
    main()
