"""Kaggle kernel + dataset automation for LunarSite.

Run ML training and evaluation on Kaggle from the terminal - no browser needed.
Wraps the Kaggle CLI so you can push notebooks, poll status, and pull outputs
without clicking through the web UI.

Usage:
    python scripts/kaggle_run.py push eval_v1_vs_v2
    python scripts/kaggle_run.py status eval_v1_vs_v2
    python scripts/kaggle_run.py wait eval_v1_vs_v2
    python scripts/kaggle_run.py pull eval_v1_vs_v2
    python scripts/kaggle_run.py run eval_v1_vs_v2    # push + wait + pull

Kernels are defined in the KERNELS dict below. To register a new kernel:
    1. Add an entry with slug, notebook path, accelerator, and datasets
    2. The metadata file is generated automatically at push time

Prerequisites:
    pip install kaggle
    Place kaggle.json at ~/.kaggle/kaggle.json (see docs: https://github.com/Kaggle/kaggle-api)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_USERNAME = "encinas88"


@dataclass
class KernelSpec:
    slug: str
    title: str
    notebook: Path
    datasets: list[str] = field(default_factory=list)
    accelerator: str = "GPU T4 x2"
    enable_gpu: bool = True
    enable_internet: bool = True
    is_private: bool = True
    language: str = "python"
    kernel_type: str = "notebook"
    output_dir: Path = field(default_factory=lambda: REPO_ROOT / "outputs")


KERNELS: dict[str, KernelSpec] = {
    "eval_v1_vs_v2": KernelSpec(
        slug=f"{KAGGLE_USERNAME}/lunarsite-eval-v1-vs-v2",
        title="LunarSite Eval v1 vs v2",
        notebook=REPO_ROOT / "notebooks" / "eval_v1_vs_v2_kaggle.ipynb",
        datasets=[
            f"{KAGGLE_USERNAME}/lunarsite-checkpoints",
            "romainpessia/artificial-lunar-rocky-landscape-dataset",
        ],
        output_dir=REPO_ROOT / "outputs" / "v1_vs_v2_eval",
    ),
}


def _find_kaggle_cli() -> str:
    """Locate kaggle CLI across common install paths on Windows/macOS/Linux."""
    cli = shutil.which("kaggle")
    if cli:
        return cli
    windows_candidates = [
        Path.home() / "AppData/Local/Packages/PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0/LocalCache/local-packages/Python313/Scripts/kaggle.exe",
        Path.home() / "AppData/Local/Programs/Python/Python313/Scripts/kaggle.exe",
        Path.home() / "AppData/Roaming/Python/Python313/Scripts/kaggle.exe",
    ]
    for p in windows_candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "Could not find kaggle CLI. Install with `pip install kaggle` and ensure it is on PATH."
    )


KAGGLE_CLI = _find_kaggle_cli()


def _run(args: list[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(args)}")
    return subprocess.run(args, cwd=str(cwd) if cwd else None, check=check, text=True)


def _build_kernel_metadata(spec: KernelSpec, staging_dir: Path) -> Path:
    """Write kernel-metadata.json + copy notebook into a staging dir for push."""
    staged_notebook = staging_dir / spec.notebook.name
    shutil.copy(spec.notebook, staged_notebook)

    metadata = {
        "id": spec.slug,
        "title": spec.title,
        "code_file": spec.notebook.name,
        "language": spec.language,
        "kernel_type": spec.kernel_type,
        "is_private": spec.is_private,
        "enable_gpu": spec.enable_gpu,
        "enable_internet": spec.enable_internet,
        "dataset_sources": spec.datasets,
        "competition_sources": [],
        "kernel_sources": [],
    }
    metadata_path = staging_dir / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata_path


def cmd_push(name: str) -> None:
    spec = KERNELS[name]
    with tempfile.TemporaryDirectory() as td:
        staging = Path(td)
        _build_kernel_metadata(spec, staging)
        _run([KAGGLE_CLI, "kernels", "push", "-p", str(staging)])
    print(f"Pushed {spec.slug}. View at https://www.kaggle.com/code/{spec.slug}")


def cmd_status(name: str) -> str:
    spec = KERNELS[name]
    result = subprocess.run(
        [KAGGLE_CLI, "kernels", "status", spec.slug],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    print(output)
    return output.lower()


def cmd_wait(name: str, poll_seconds: int = 60) -> str:
    """Poll until the kernel finishes running. Returns final status."""
    spec = KERNELS[name]
    terminal = {"complete", "error", "cancelAcknowledged", "cancelled"}
    start = time.time()
    while True:
        result = subprocess.run(
            [KAGGLE_CLI, "kernels", "status", spec.slug],
            check=True,
            capture_output=True,
            text=True,
        )
        status_line = result.stdout.strip()
        elapsed_m = (time.time() - start) / 60
        print(f"[{elapsed_m:6.1f}m] {status_line}")
        low = status_line.lower()
        if any(t.lower() in low for t in terminal):
            return status_line
        time.sleep(poll_seconds)


def cmd_pull(name: str) -> None:
    spec = KERNELS[name]
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    _run([KAGGLE_CLI, "kernels", "output", spec.slug, "-p", str(spec.output_dir)])
    print(f"Outputs pulled to {spec.output_dir}")


def cmd_run(name: str) -> None:
    """Full loop: push, wait for completion, pull outputs."""
    cmd_push(name)
    print("Waiting for kernel to complete (polling every 60s)...")
    final = cmd_wait(name)
    if "error" in final.lower():
        print(f"Kernel finished with error: {final}")
        sys.exit(1)
    cmd_pull(name)
    print(f"Done. Outputs at {KERNELS[name].output_dir}")


def cmd_list_kernels(_: str) -> None:
    for name, spec in KERNELS.items():
        print(f"{name}: {spec.slug}")
        print(f"  notebook: {spec.notebook.relative_to(REPO_ROOT)}")
        print(f"  datasets: {spec.datasets}")
        print(f"  output:   {spec.output_dir.relative_to(REPO_ROOT)}")


COMMANDS = {
    "push": cmd_push,
    "status": cmd_status,
    "wait": cmd_wait,
    "pull": cmd_pull,
    "run": cmd_run,
    "list": cmd_list_kernels,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle kernel automation for LunarSite")
    parser.add_argument("command", choices=list(COMMANDS.keys()))
    parser.add_argument("name", nargs="?", default="", help="Kernel name from KERNELS dict")
    args = parser.parse_args()

    if args.command != "list" and args.name not in KERNELS:
        print(f"Unknown kernel '{args.name}'. Available: {list(KERNELS.keys())}")
        sys.exit(1)

    COMMANDS[args.command](args.name)


if __name__ == "__main__":
    main()
