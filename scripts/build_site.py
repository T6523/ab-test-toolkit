"""Rebuild the demo page's data: the abkit wheel and the precomputed default results.

Run from the repo root after changing abkit or docs/glue.py (needs `pip install build playwright`
and `playwright install chromium`):
    python scripts/build_site.py

The defaults are computed inside the page's own Pyodide (headless Chromium), because the browser's NumPy can draw
slightly different random numbers from the same seed than a desktop NumPy. That keeps the cached numbers identical
to what the page shows after you press a button. Without playwright it falls back to native Python and warns.
"""
import functools
import json
import runpy
import shutil
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
FUNC = {"plan": "plan", "analyze": "analyze", "peek": "peeking", "path": "path", "low": "lowpower"}


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def compute_in_browser(defaults):
    from playwright.sync_api import sync_playwright
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(DOCS)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{server.server_address[1]}/index.html")
            page.wait_for_function("window.__ready === true", timeout=300_000)
            out = {key: json.loads(page.evaluate("([fn, args]) => py.globals.get(fn)(...args)", [FUNC[key], list(args)]))
                   for key, args in defaults.items()}
            browser.close()
        return out
    finally:
        server.shutdown()


def compute_native(glue, defaults):
    print("WARNING: playwright not available; computing with desktop NumPy. Simulation numbers may differ slightly "
          "from the browser's.", file=sys.stderr)
    return {key: json.loads(glue[FUNC[key]](*args)) for key, args in defaults.items()}


def main():
    subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(DOCS), str(ROOT)], check=True)
    shutil.rmtree(ROOT / "build", ignore_errors=True)
    glue = runpy.run_path(str(DOCS / "glue.py"))
    defaults = glue["DEFAULTS"]
    try:
        results = compute_in_browser(defaults)
    except ImportError:
        results = compute_native(glue, defaults)
    (DOCS / "initial.json").write_text(json.dumps({"args": {k: list(v) for k, v in defaults.items()}, **results}))
    print(f"wrote {DOCS / 'initial.json'}")


if __name__ == "__main__":
    main()
