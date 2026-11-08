"""Measure unmodified SGLang baselines on H200 using the pinned Muse Spark runtime."""

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
REMOTE = "/root/okc-muse-spark"
SGLANG_COMMIT = "a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f"
# Official dev-muse-spark-open image, pinned to the linux/amd64 manifest.
SGLANG_IMAGE = "lmsysorg/sglang@sha256:a0ae9cefecc0fb68e74a6f3b7f338eb80a53e3e6df4637ec2f024abc5fec07af"
image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    # Retain native dependencies from the official image, but pin the Python/JIT
    # sources to the exact revision named in every supplied baseline.py.
    .run_commands(
        f"curl -fL https://github.com/sgl-project/sglang/archive/{SGLANG_COMMIT}.tar.gz -o /tmp/sglang-pinned.tar.gz",
        "mkdir -p /tmp/sglang-pinned",
        "tar -xzf /tmp/sglang-pinned.tar.gz --strip-components=1 -C /tmp/sglang-pinned",
        "cp -a /tmp/sglang-pinned/python/sglang/. /sgl-workspace/sglang-muse-spark/python/sglang/",
        "rm -rf /tmp/sglang-pinned /tmp/sglang-pinned.tar.gz",
    )
    .env({"TORCH_CUDA_ARCH_LIST": "9.0", "PYTHONUNBUFFERED": "1",
          "MUSE_SGLANG_COMMIT": SGLANG_COMMIT, "MUSE_SGLANG_IMAGE": SGLANG_IMAGE})
    .add_local_dir(ROOT, REMOTE, ignore=["**/__pycache__/**", "**/profiles/**"])
)
app = modal.App("okc-muse-spark-baseline-h200")


@app.function(image=image, gpu="H200", cpu=8, memory=65536, timeout=3600,
              max_containers=1, scaledown_window=30)
def evaluate(options, tool="bench.py", tool_args=()):
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "result.json"
        command = [sys.executable, f"{REMOTE}/{tool}", "--result", str(result_path), *tool_args]
        for key, value in options.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        completed = subprocess.run(command, cwd=REMOTE, check=False)
        result = (json.loads(result_path.read_text()) if result_path.exists()
                  else {"success": False, "error": "Worker exited without a report"})
        result["exit_code"] = completed.returncode
        return result


@app.local_entrypoint()
def main(problem: str = "all", suite: str = "scored", mode: str = "bench",
         repeats: int = 5, seed: int = 2026, case_id: int = -1,
         submission: str = "baseline.py", against: str = "", tool: str = "", tool_args: str = ""):
    """submission: file to time (default the SGLang baseline). against: a second file timed
    alternately on the same inputs, e.g. --submission versions/submission_triton.py --against baseline.py.
    tool: run another script in this folder instead (e.g. tune_bf16_proj.py) with tool_args."""
    if tool:
        if not (ROOT / tool).is_file() or Path(tool).name != tool:
            raise ValueError("tool must be a script directly in okc-muse-spark")
        result = evaluate.remote({}, tool, tool_args.split())
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = ROOT.parent / "modal_out" / "H200" / "tools" / f"{stamp}-{Path(tool).stem}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved: {path}")
        if result.get("exit_code") != 0:
            raise RuntimeError(f"{tool} failed; see the remote log")
        return
    spec = importlib.util.spec_from_file_location("muse_bench", ROOT / "bench.py")
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    problems = bench.PROBLEMS if problem == "all" else tuple(p.strip() for p in problem.split(","))
    jobs = [dict(problem=p, mode=mode, suite=suite, submission=submission,
                 case_id=case_id, repeats=repeats, seed=seed, against=against) for p in problems]
    for job in jobs:
        bench.validate_options(ROOT, **job)
    failures = []
    for job in jobs:
        result = evaluate.remote(job)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        tag = "baseline" if submission == "baseline.py" else Path(submission).stem
        if against:
            tag += "-vs-" + Path(against).stem
        path = ROOT.parent / "modal_out" / "H200" / job["problem"] / f"{stamp}-{tag}-{mode}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved: {path}")
        if not result.get("success") or result.get("exit_code") != 0:
            failures.append(job["problem"])
            print(f"FAILED: {result.get('error')}")
        elif "speedup" in result:
            print(f"{job['problem']}: {result['aggregate_paired']['geomean_ms']:.4f} ms vs "
                  f"{result['aggregate_against']['geomean_ms']:.4f} ms, speedup {result['speedup']:.3f}x")
        elif "aggregate" in result:
            print(f"{job['problem']}: {result['aggregate']['geomean_ms']:.4f} ms")
    if failures:
        raise RuntimeError(f"Failed baselines: {', '.join(failures)}; see saved reports")
