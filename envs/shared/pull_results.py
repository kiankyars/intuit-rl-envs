"""Pull JSON outputs from the `intuit-rl-envs-results` Modal Volume into the
local site/public/data tree.

Shells out to `modal volume get` because the direct Python `Volume.read_file`
iterator was producing truncated/stale local copies of recently-written files.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CHAPTERS = ("gameability", "verifiability", "shape", "horizon", "escalation", "reversibility")
VARIANTS = ("leaky", "patched")


def pull_one(volume: str, chapter: str, variant: str, dest: Path) -> bool:
    remote = f"/{chapter}/{variant}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["modal", "volume", "get", "--force", volume, remote, str(dest)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(f"  skip {chapter}/{variant}: {proc.stderr.strip().splitlines()[-1] if proc.stderr else 'unknown error'}\n")
        return False
    size = dest.stat().st_size if dest.exists() else 0
    print(f"  wrote {dest} ({size} bytes)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapter", choices=CHAPTERS, default=None,
                    help="only pull this chapter; default = all")
    ap.add_argument("--variant", choices=VARIANTS, default=None,
                    help="only pull this variant; default = all")
    ap.add_argument("--volume", default="intuit-rl-envs-results")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    data = root / "site" / "public" / "data"

    chapters = (args.chapter,) if args.chapter else CHAPTERS
    variants = (args.variant,) if args.variant else VARIANTS

    pulled = 0
    for ch in chapters:
        for var in variants:
            if pull_one(args.volume, ch, var, data / ch / f"{var}.json"):
                pulled += 1

    print(f"\npulled {pulled} file(s)")
    return 0 if pulled else 1


if __name__ == "__main__":
    raise SystemExit(main())
