"""Modal app for running any chapter's train.py on a 1×H100.

Usage (after `pip install modal` and `modal token new`):

    modal run envs/shared/modal_app.py::train --chapter gameability --variant leaky
    modal run envs/shared/modal_app.py::train --chapter gameability --variant patched

Modal mounts the whole repo so the chapter's train.py + envs/shared/ are
available. The written JSON is downloaded back into site/public/data/.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.48",
        "trl>=0.14",
        "accelerate>=1.2",
        "datasets>=3.0",
        "vllm>=0.7",
        "outlines>=0.1",
        "sentencepiece",
    )
    .add_local_dir(str(ROOT), "/repo")
)

app = modal.App("intuit-rl-envs")

volume = modal.Volume.from_name("intuit-rl-envs-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": volume},
)
def train_remote(chapter: str, variant: str) -> bytes:
    import importlib
    import sys

    sys.path.insert(0, "/repo")
    os.chdir("/repo")
    mod = importlib.import_module(f"envs.{chapter}.train")
    path = mod.run(variant=variant)
    return Path(path).read_bytes()


@app.local_entrypoint()
def train(chapter: str, variant: str = "leaky"):
    payload = train_remote.remote(chapter=chapter, variant=variant)
    out = ROOT / "site" / "public" / "data" / chapter / f"{variant}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    print(f"wrote {out.relative_to(ROOT)} ({len(payload)} bytes)")
