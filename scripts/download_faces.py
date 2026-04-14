"""Download a small curated set of public face images into data/source_images/.

We pull from two well-known OSS repositories that ship face images as test
fixtures:

  - ageitgey/face_recognition (examples/)
  - davisking/dlib (examples/faces/)

These are redistributable test images that have been used in countless face-
detection tutorials. For this project we only need a modest number of faces
to seed the synthetic fake-video generator (we apply affine spoofs to them).

Usage:
    python -m scripts.download_faces
    python -m scripts.download_faces --out data/source_images --force
"""
from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from pathlib import Path

# (url, destination filename) — destination names are normalized/prefixed so
# they remain unique across sources.
SOURCES: list[tuple[str, str]] = [
    # face_recognition examples — single-face photos of public figures
    (
        "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg",
        "fr_obama.jpg",
    ),
    (
        "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama2.jpg",
        "fr_obama2.jpg",
    ),
    (
        "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg",
        "fr_biden.jpg",
    ),
    (
        "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/alex-lacamoire.png",
        "fr_alex.png",
    ),
    (
        "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/lin-manuel-miranda.png",
        "fr_miranda.png",
    ),
    # dlib face examples — PASCAL VOC-origin photos, many with a single face
    (
        "https://raw.githubusercontent.com/davisking/dlib/master/examples/faces/2007_007763.jpg",
        "dlib_2007_007763.jpg",
    ),
    (
        "https://raw.githubusercontent.com/davisking/dlib/master/examples/faces/2008_001009.jpg",
        "dlib_2008_001009.jpg",
    ),
    (
        "https://raw.githubusercontent.com/davisking/dlib/master/examples/faces/2008_001322.jpg",
        "dlib_2008_001322.jpg",
    ),
    (
        "https://raw.githubusercontent.com/davisking/dlib/master/examples/faces/2008_002079.jpg",
        "dlib_2008_002079.jpg",
    ),
    (
        "https://raw.githubusercontent.com/davisking/dlib/master/examples/faces/2008_002470.jpg",
        "dlib_2008_002470.jpg",
    ),
    (
        "https://raw.githubusercontent.com/davisking/dlib/master/examples/faces/2008_002506.jpg",
        "dlib_2008_002506.jpg",
    ),
    (
        "https://raw.githubusercontent.com/davisking/dlib/master/examples/faces/2008_004176.jpg",
        "dlib_2008_004176.jpg",
    ),
    (
        "https://raw.githubusercontent.com/davisking/dlib/master/examples/faces/2008_007676.jpg",
        "dlib_2008_007676.jpg",
    ),
    (
        "https://raw.githubusercontent.com/davisking/dlib/master/examples/faces/2009_004587.jpg",
        "dlib_2009_004587.jpg",
    ),
    (
        "https://raw.githubusercontent.com/davisking/dlib/master/examples/faces/Tom_Cruise_avp_2014_4.jpg",
        "dlib_tom_cruise.jpg",
    ),
]


def _download(url: str, dest: Path, timeout: int = 30) -> bool:
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "sys-analytics/0.1"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"  [fail] {url}: {e}")
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out", type=Path, default=Path("data/source_images/faces")
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if the destination file already exists",
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ok = 0
    skipped = 0
    for url, name in SOURCES:
        dest = args.out / name
        if dest.exists() and not args.force:
            skipped += 1
            continue
        print(f"  downloading {name} ...")
        if _download(url, dest):
            ok += 1

    existing = sum(1 for _, name in SOURCES if (args.out / name).exists())
    print(
        f"\ndownloaded: {ok}, already present: {skipped}, "
        f"total on disk: {existing}/{len(SOURCES)}"
    )
    print(f"destination: {args.out.resolve()}")


if __name__ == "__main__":
    main()
