"""H200 development runner. See MODAL.md for checks, timings and official submission."""

import json
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
REMOTE = "/root/okc-muse-spark"

# CUDA devel also supports torch.utils.cpp_extension.load_inline / nvcc.
# Keep the Torch/CUDA versions consistent with this repository's modal_run.py.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.13"
    )
    .apt_install("build-essential", "ninja-build")
    .pip_install("torch==2.13.0", "triton==3.7.1", "numpy", "packaging", "ninja")
    .env({"TORCH_CUDA_ARCH_LIST": "9.0", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(
        ROOT, REMOTE,
        ignore=["**/__pycache__/**", "**/profiles/**", "**/.git/**", "**/.venv/**"],
    )
)
app = modal.App("okc-muse-spark-h200")


@app.function(image=image, gpu="H200", cpu=8, memory=32768, timeout=1800,
              max_containers=1, scaledown_window=2)
def evaluate(options: dict) -> dict:
    # Each invocation gets a fresh interpreter: same-named checks/nvfp4 modules
    # from different problems must never leak into the next run.
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "result.json"
        command = [sys.executable, f"{REMOTE}/bench.py", "--result", str(result_path)]
        for key, value in options.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        completed = subprocess.run(command, cwd=REMOTE, check=False)
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            result = {"success": False, "error": "Worker exited before writing its report"}
        result["exit_code"] = completed.returncode
        return result


@app.local_entrypoint()
def main(
    problem: str = "muse_bf16_proj",
    mode: str = "bench",
    suite: str = "smoke",
    submission: str = "submission.py",
    case_id: int = -1,
    repeats: int = 5,
    seed: int = 2026,
):
    """mode: check/bench; suite: smoke/tests/scored; case-id: -1 means all."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("muse_bench", ROOT / "bench.py")
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    options = dict(problem=problem, mode=mode, suite=suite, submission=submission,
                   case_id=case_id, repeats=repeats, seed=seed)
    bench.validate_options(ROOT, **options)
    result = evaluate.remote(options)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT.parent / "modal_out" / "H200" / problem / f"{stamp}-{mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved: {output}")
    if not result.get("success") or result.get("exit_code") != 0:
        raise RuntimeError(f"H200 {mode} failed; see {output} and the remote log")
