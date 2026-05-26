"""Modal app for running any chapter's train.py on a 1×H100.

Usage:

    modal run envs/shared/modal_app.py::train --chapter gameability --variant leaky
    modal run envs/shared/modal_app.py::train --chapter gameability --variant patched

Each run also persists the JSON it produces into the `intuit-rl-envs-results`
Modal Volume, keyed by chapter/variant, so the artifact is recoverable even if
the local entrypoint disconnects (e.g. when launched with --detach).
"""

from __future__ import annotations

from pathlib import Path

import modal

# Local repo root for `add_local_dir` + writing returned bytes back to disk.
# Resolved relative to this file's filesystem location at LOCAL definition
# time only; the `modal_app.py` file inside the container does not need it.
_here = Path(__file__).resolve()
LOCAL_ROOT = _here.parents[2] if len(_here.parents) >= 3 else Path("/repo")
REMOTE_ROOT = "/repo"

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
    .add_local_dir(
        str(LOCAL_ROOT),
        REMOTE_ROOT,
        ignore=[
            '**/node_modules/**',
            '**/dist/**',
            '**/.astro/**',
            '**/__pycache__/**',
            '**/_runs/**',
            '.git/**',
            '.venv/**',
        ],
    )
)

app = modal.App("intuit-rl-envs")

hf_cache = modal.Volume.from_name("intuit-rl-envs-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("intuit-rl-envs-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 2,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/results": results,
    },
)
def train_remote(chapter: str, variant: str) -> bytes:
    import importlib
    import os
    import shutil
    import sys
    import traceback

    sys.path.insert(0, REMOTE_ROOT)
    os.chdir(REMOTE_ROOT)
    try:
        mod = importlib.import_module(f"envs.{chapter}.train")
        path = mod.run(variant=variant)
    except Exception:
        traceback.print_exc()
        raise

    # Persist a copy to the results Volume for recovery.
    dest_dir = Path("/results") / chapter
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{variant}.json"
    shutil.copy(path, dest)
    results.commit()
    print(f"persisted {dest} ({dest.stat().st_size} bytes)")

    return Path(path).read_bytes()


@app.local_entrypoint()
def train(chapter: str, variant: str = "leaky"):
    payload = train_remote.remote(chapter=chapter, variant=variant)
    out = LOCAL_ROOT / "site" / "public" / "data" / chapter / f"{variant}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    print(f"wrote {out.relative_to(LOCAL_ROOT)} ({len(payload)} bytes)")
