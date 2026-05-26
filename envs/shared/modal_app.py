"""Modal app for running any chapter's train.py on a 1×H100.

Usage:

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

# Pinned to TRL 1.5.x. TRL 1.5 declares vllm 0.12–0.18 and transformers >=4.56.2.
# torch is pulled in by vllm and matches its preferred CUDA build.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "trl[vllm]==1.5.0",
        "accelerate>=1.4.0",
        "datasets>=4.7.0",
        "transformers>=4.56.2,!=5.1.0",
        "vllm>=0.12.0,<=0.18.0",
        "sentencepiece",
        "hf-transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(str(ROOT), "/repo")
)

app = modal.App("intuit-rl-envs")

hf_cache = modal.Volume.from_name("intuit-rl-envs-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def train_remote(chapter: str, variant: str) -> bytes:
    import importlib
    import sys
    import traceback

    sys.path.insert(0, "/repo")
    os.chdir("/repo")
    try:
        mod = importlib.import_module(f"envs.{chapter}.train")
        path = mod.run(variant=variant)
    except Exception:
        traceback.print_exc()
        raise
    return Path(path).read_bytes()


@app.local_entrypoint()
def train(chapter: str, variant: str = "leaky"):
    payload = train_remote.remote(chapter=chapter, variant=variant)
    out = ROOT / "site" / "public" / "data" / chapter / f"{variant}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    print(f"wrote {out.relative_to(ROOT)} ({len(payload)} bytes)")
