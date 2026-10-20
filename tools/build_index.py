"""Wrap src/page.html (head tags, styles, body content) into a standalone index.html for static hosting.

src/page.html is also what gets published as the Artifact preview, which adds its own document skeleton,
so it carries no <!doctype>/<html>/<head>/<body> of its own. This adds them, plus the link-preview tags.

    python3 tools/build_index.py [--url https://example.org/diplo/]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TITLE = "DiPLO: Self-distilled Diffusion Planning for Loco-manipulation"
DESCRIPTION = (
    "A diffusion planner generates whole-body robot and object motion, an RL tracking policy executes it, and two "
    "post-training stages (SD:DPO, DA:PPO) remove the need for human-motion conditioning. Deployed on a Unitree G1."
)
FAVICON = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' rx='7' fill='%23060a16'/%3E"
    "%3Cpath d='M6 12V6h6M20 6h6v6M26 20v6h-6M12 26H6v-6' fill='none' stroke='%2359d9ff' stroke-width='2.4'/%3E"
    "%3Crect x='12' y='12' width='8' height='8' rx='1.5' fill='%23e8eefb'/%3E%3C/svg%3E"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="", help="public URL of the site, for absolute link-preview image URLs")
    args = ap.parse_args()

    page = (ROOT / "src" / "page.html").read_text()
    # The <title> and description in page.html are the Artifact's; the standalone page gets the full ones below.
    page = re.sub(r"<title>.*?</title>\n", "", page, count=1)
    page = re.sub(r'<meta name="description"[^>]*>\n', "", page, count=1)
    page = page.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")
    head_end = page.index("</style>") + len("</style>")
    head, body = page[:head_end], page[head_end:]

    base = args.url.rstrip("/") + "/" if args.url else ""
    meta = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{TITLE}</title>
<meta name="description" content="{DESCRIPTION}">
<meta name="theme-color" content="#060a16">
<meta property="og:type" content="website">
<meta property="og:title" content="{TITLE}">
<meta property="og:description" content="{DESCRIPTION}">
<meta property="og:image" content="{base}static/img/og.jpg">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
{f'<meta property="og:url" content="{base}">' if base else ""}
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{TITLE}">
<meta name="twitter:description" content="{DESCRIPTION}">
<meta name="twitter:image" content="{base}static/img/og.jpg">
<link rel="icon" href="{FAVICON}">
"""
    out = meta + head.strip() + "\n</head>\n<body>\n" + body.strip() + "\n</body>\n</html>\n"
    (ROOT / "index.html").write_text(out)
    print(f"wrote {ROOT / 'index.html'} ({len(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
