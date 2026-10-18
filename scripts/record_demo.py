"""Record assets/demo.gif from the live demo page (headless Chromium plus ffmpeg).

Run from the repo root:  python scripts/record_demo.py [url]
Needs playwright (with chromium) and ffmpeg. The recording is the real page doing real work, no edited frames.
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
URL = sys.argv[1] if len(sys.argv) > 1 else "https://t6523.github.io/ab-test-toolkit/"
VIEW = {"width": 1000, "height": 600}
WIDTH, FPS = 640, 8   # gif width in pixels and frames per second, tuned to stay under 5 MB
OUT = ROOT / "assets" / "demo.gif"


def scroll_to(page, selector, offset=0, wait=1700):
    page.evaluate("([s, o]) => window.scrollTo({ top: document.querySelector(s).getBoundingClientRect().top + scrollY - o, behavior: 'smooth' })",
                  [selector, offset])
    page.wait_for_timeout(wait)


def record(tmp):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport=VIEW, record_video_dir=str(tmp), record_video_size=VIEW, color_scheme="light")
        page = ctx.new_page()
        t0 = time.monotonic()
        page.goto(URL)
        page.wait_for_function("window.__ready === true", timeout=240_000)   # Pyodide and abkit are loaded
        ready_at = time.monotonic() - t0
        crop = page.locator("main").bounding_box()
        page.wait_for_timeout(900)

        # 1. type a new input at the top and show the numbers update
        page.click("#plan-form [name=mde]", click_count=3)
        page.keyboard.type("0.5", delay=160)
        page.wait_for_timeout(400)
        page.click("#plan-form button")
        page.wait_for_timeout(1500)

        # 2. scroll through the graphs and stop at each one
        scroll_to(page, "#plan-result", offset=20)       # power against sample size
        page.wait_for_timeout(1800)
        scroll_to(page, "#analyze-result", offset=20)    # interval and verdict
        page.wait_for_timeout(1800)
        scroll_to(page, "#peek-chart", offset=70)        # peeking against sequential false positives
        page.wait_for_timeout(1800)
        scroll_to(page, "#path-fooled", offset=20)       # one test, look by look
        page.click("#path-fooled")
        page.wait_for_timeout(2200)
        scroll_to(page, "#low-result", offset=20)        # low power exaggeration
        page.wait_for_timeout(1800)

        video = Path(page.video.path())
        ctx.close()
        browser.close()
    return video, ready_at, crop


def to_gif(video, start, duration, crop):
    x, y, w, h = (int(crop[k]) // 2 * 2 for k in ("x", "y", "width", "height"))
    h = min(h, VIEW["height"]) // 2 * 2
    vf = f"crop={w}:{h}:{x}:0,fps={FPS},scale={WIDTH}:-1:flags=lanczos"
    with tempfile.TemporaryDirectory() as d:
        palette = Path(d) / "palette.png"
        base = ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-t", f"{duration:.2f}", "-i", str(video)]
        subprocess.run(base + ["-vf", vf + ",palettegen=stats_mode=diff", str(palette)], check=True, capture_output=True)
        subprocess.run(base + ["-i", str(palette), "-lavfi", f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                               str(OUT)], check=True, capture_output=True)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        video, ready_at, crop = record(Path(tmp))
        total = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                                     capture_output=True, text=True, check=True).stdout)
        start = max(ready_at - 0.3, 0)   # trim the wait for Python to load, start when the page says Ready
        OUT.parent.mkdir(exist_ok=True)
        to_gif(video, start, total - start, crop)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB), page ready after {ready_at:.1f}s, gif {total - start:.1f}s")


if __name__ == "__main__":
    main()
