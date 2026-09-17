"""Fetch the public sample images used by this project into data/public/.

Why this script exists: `.gitignore` excludes `data/public/*`, so the sample
images are intentionally NOT versioned.  After cloning this project on a new
machine, run this script to restore them.

Usage
-----
    python fetch_public_samples.py                  # 5 images from Wikimedia Commons (~5 MB)
    python fetch_public_samples.py --with-docunet   # + 4 images from the DocUNet benchmark
                                                    #   (downloads a 344 MB zip once, then
                                                    #    extracts only the needed files)
    python fetch_public_samples.py --out-dir data/public --max-side 1600

Licensing
---------
Every image is credited in `data/public/SOURCES.md`.  If you use the DocUNet
samples in written work you must cite Ma et al., CVPR 2018.
"""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

USER_AGENT = "ComputerVision-coursework/1.0 (educational use; homography assignment)"

COMMONS_FILES = [
    ("Chessboard 2.jpg", "chessboard"),
    ("Book-cover.jpg", "book_cover"),
    ("Flipchart Motivation03.jpg", "flipchart_venn"),
    ("Flipchart ToDo-Liste.jpg", "flipchart_table"),
    ("A black menu board from Kau Yee Hong Kong style restaurant.jpg", "menu_board"),
]

DOCUNET_ZIP = "http://vision.cs.stonybrook.edu/~kema/docwarp/original.zip"
DOCUNET_MEMBERS = ["30_1.jpg", "30_2.jpg", "29_1.jpg", "29_2.jpg"]


def download(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def writable_image(path: Path, image: np.ndarray, max_side: int) -> tuple[int, int, int]:
    """Downscale if needed, write JPEG, return (width, height, quality_note)."""
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return image.shape[1], image.shape[0], image.shape[1] * image.shape[0]


def fetch_commons(out_dir: Path, max_side: int) -> int:
    done = 0
    for commons_name, stem in COMMONS_FILES:
        dest = out_dir / f"{stem}.jpg"
        url = ("https://commons.wikimedia.org/wiki/Special:FilePath/"
               + urllib.parse.quote(commons_name) + f"?width={max_side}")
        try:
            raw = download(url)
        except Exception as exc:
            print(f"  FAIL {stem:<18} {exc}")
            continue
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  FAIL {stem:<18} could not decode downloaded bytes")
            continue
        width, height, _ = writable_image(dest, image, max_side)
        print(f"  ok   {dest}  {width}x{height}")
        done += 1
    return done


def fetch_docunet(out_dir: Path, max_side: int) -> int:
    with tempfile.TemporaryDirectory(prefix="docunet-") as tmp:
        zip_path = Path(tmp) / "original.zip"
        print(f"  downloading {DOCUNET_ZIP}")
        print("  (344 MB, one time -- this may take a few minutes)")

        def reporthook(block: int, block_size: int, total: int) -> None:
            if total > 0:
                percent = min(100.0, block * block_size * 100.0 / total)
                print(f"\r  {percent:5.1f}%", end="", flush=True)

        urllib.request.urlretrieve(DOCUNET_ZIP, zip_path, reporthook)
        print()

        done = 0
        with zipfile.ZipFile(zip_path) as archive:
            available = {Path(name).name: name for name in archive.namelist()}
            for member in DOCUNET_MEMBERS:
                if member not in available:
                    print(f"  FAIL {member}: not present in archive")
                    continue
                payload = archive.read(available[member])
                image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    print(f"  FAIL {member}: could not decode")
                    continue
                stem = f"docunet_{Path(member).stem.replace('_', '_view')}"
                dest = out_dir / f"{stem}.jpg"
                width, height, _ = writable_image(dest, image, max_side)
                print(f"  ok   {dest}  {width}x{height}")
                done += 1
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore data/public sample images")
    parser.add_argument("--out-dir", type=Path, default=Path("data/public"))
    parser.add_argument("--max-side", type=int, default=1600,
                        help="longest edge in pixels; the assignment allows lowering resolution")
    parser.add_argument("--with-docunet", action="store_true",
                        help="also fetch 4 images from the DocUNet benchmark (344 MB download)")
    args = parser.parse_args()

    total = 0
    print("Wikimedia Commons:")
    total += fetch_commons(args.out_dir, args.max_side)

    if args.with_docunet:
        print("DocUNet benchmark:")
        total += fetch_docunet(args.out_dir, args.max_side)
    else:
        print("DocUNet: skipped (pass --with-docunet to fetch)")

    print(f"\nfetched {total} image(s) into {args.out_dir}")
    print("see data/public/SOURCES.md for attribution and licences")
    if total == 0:
        print("nothing fetched -- check your network connection")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
