"""Pull JSON outputs from the `intuit-rl-envs-results` Modal Volume into the
local site/public/data tree.

Detached `modal run --detach …` jobs persist their JSON to the Volume rather
than streaming bytes back to the local entrypoint, so we re-hydrate the local
copy from the Volume on demand. Run after a detached batch finishes:

    python -m envs.shared.pull_results
    python -m envs.shared.pull_results --chapter gameability
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import modal


CHAPTERS = ("gameability", "verifiability", "shape", "horizon", "escalation", "reversibility")
VARIANTS = ("leaky", "patched")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapter", choices=CHAPTERS, default=None,
                    help="only pull this chapter; default = all")
    ap.add_argument("--volume", default="intuit-rl-envs-results")
    args = ap.parse_args()

    vol = modal.Volume.from_name(args.volume)
    root = Path(__file__).resolve().parents[2]
    data = root / "site" / "public" / "data"

    chapters = (args.chapter,) if args.chapter else CHAPTERS
    pulled = 0
    for chapter in chapters:
        out_dir = data / chapter
        out_dir.mkdir(parents=True, exist_ok=True)
        for variant in VARIANTS:
            remote = f"/{chapter}/{variant}.json"
            local = out_dir / f"{variant}.json"
            try:
                buf = io.BytesIO()
                for chunk in vol.read_file(remote.lstrip("/")):
                    buf.write(chunk)
                payload = buf.getvalue()
            except Exception as e:
                print(f"  skip {chapter}/{variant}: {e.__class__.__name__}: {e}", file=sys.stderr)
                continue
            local.write_bytes(payload)
            print(f"  wrote {local.relative_to(root)} ({len(payload)} bytes)")
            pulled += 1

    print(f"\npulled {pulled} file(s)")
    return 0 if pulled else 1


if __name__ == "__main__":
    raise SystemExit(main())
