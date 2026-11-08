"""Standalone correctness/CUDA-graph runner for the supplied OKC problem interfaces.

GPU MODE's kernelbot runner/evaluator informed the container and check/bench split:
https://github.com/gpu-mode/kernelbot/tree/main/src/runners
https://github.com/gpu-mode/kernelbot/blob/main/examples/eval.py
This is a development measurement, not the official OKC judge or score.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROBLEMS = (
    "muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16",
    "muse_decode_layer", "muse_prefill_layer",
)


def parse_shape(text):
    fields = {}
    for part in text.split(","):
        key, value = part.strip().split(maxsplit=1)
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        fields[key] = value
    return fields


def cases_for(root, problem, suite):
    """Use the user's TASK.md as the source of truth for public case shapes."""
    text = (root / problem / "TASK.md").read_text()
    if suite in ("tests", "smoke"):
        match = re.search(r"Before timing,.*?```\s*\n(.*?)```", text, re.S)
        if not match:
            raise ValueError("Cannot find the public tests in TASK.md")
        cases = [{"shape": parse_shape(line), "label": line.strip()}
                 for line in match[1].splitlines() if line.strip()]
        return cases[:1] if suite == "smoke" else cases
    if suite == "scored":
        cases = []
        for line in text.splitlines():
            if not line.startswith("|"):
                continue
            cols = [v.strip() for v in line.strip("|").split("|")]
            if len(cols) == 4 and cols[2].isdigit():
                cases.append({"shape": parse_shape(cols[1]), "label": cols[1],
                              "load": cols[0], "weight": int(cols[2])})
        if not cases:
            raise ValueError("Cannot find scored shapes in TASK.md")
        return cases
    raise ValueError(f"Unknown suite: {suite}")


def validate_options(root, problem, mode, suite, submission, case_id, repeats, seed, against=""):
    if problem not in PROBLEMS:
        raise ValueError(f"Unknown problem: {problem}; choose from {PROBLEMS}")
    if mode not in ("check", "bench"):
        raise ValueError("mode must be check or bench")
    if repeats < 1 or not 0 <= seed < 2**63:
        raise ValueError("repeats must be positive; seed must be in [0, 2**63)")
    folder = (root / problem).resolve()
    script = (folder / submission).resolve()
    for name in (submission, against) if against else (submission,):
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or path.suffix != ".py" or not path.is_file():
            raise ValueError(f"{name} must be an existing .py file within the problem folder")
    cases = cases_for(root, problem, suite)
    if case_id < -1 or case_id >= len(cases):
        raise ValueError(f"case-id must be -1 or 0..{len(cases) - 1}")
    return script, [(i, c) for i, c in enumerate(cases) if case_id in (-1, i)]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clone_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_tree(v) for v in value)
    return value


def copy_tree(target, source):
    """Update graph inputs in place; reject changes that would invalidate capture."""
    import torch
    if isinstance(target, torch.Tensor):
        if not isinstance(source, torch.Tensor) or (
            target.shape, target.dtype, target.device
        ) != (source.shape, source.dtype, source.device):
            raise ValueError("Fresh input tensor layout differs from captured input")
        target.copy_(source)
    elif isinstance(target, dict):
        if not isinstance(source, dict) or target.keys() != source.keys():
            raise ValueError("Fresh input dictionary differs from captured input")
        for key in target:
            copy_tree(target[key], source[key])
    elif isinstance(target, (tuple, list)):
        if type(target) is not type(source) or len(target) != len(source):
            raise ValueError("Fresh input sequence differs from captured input")
        for dst, src in zip(target, source):
            copy_tree(dst, src)
    elif target != source:
        raise ValueError(f"Non-tensor graph input changed: {target!r} -> {source!r}")


def make_input(reference, shape, seed, weights):
    args = {**shape, "seed": seed, "device": "cuda"}
    if hasattr(reference, "case_fields"):
        args.update(reference.case_fields(args, weights))
    return reference.generate_input(**args)


def verify(reference, original, actual_data, output, weights):
    # Attention/layer checks need the newly written KV entries as well as output.
    raw = reference.collect(actual_data, output) if hasattr(reference, "collect") else output
    args = (original, raw, weights) if weights is not None else (original, raw)
    ok, message = reference.check_implementation(*args)
    if not ok:
        raise AssertionError(message)
    return message


def prepare_graph(reference, kernel, state, weights, original):
    """Check one eager call, warm up, and capture flush + timed kernel as a CUDA graph."""
    import torch

    def invoke(data):
        return kernel(data, state) if weights is not None else kernel(data)

    static = clone_tree(original)
    output = invoke(static)
    torch.cuda.synchronize()
    message = verify(reference, original, static, output, weights)
    del output

    # Warm up outside capture, including compilation and library initialization.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            copy_tree(static, original)
            invoke(static)
    stream.synchronize()

    # Read 128 MiB (larger than H200 L2) before EVERY timed kernel invocation.
    # Events are captured in the graph; neither cache flushing nor host enqueue
    # overhead is included in the reported device interval.
    flush = torch.ones(32 * 1024 * 1024, dtype=torch.float32, device="cuda")
    flush_sum = torch.empty((), dtype=torch.float32, device="cuda")
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        torch.sum(flush, dim=0, out=flush_sum)
        start.record()
        output = invoke(static)
        end.record()
    # Everything the graph writes or reads must outlive it: a freed flush_sum would be
    # handed to the next input tensor and overwritten by every replay.
    return {"static": static, "output": output, "graph": graph, "start": start,
            "end": end, "flush": flush, "flush_sum": flush_sum, "message": message}


def replay_us(captured):
    import torch
    torch.cuda.synchronize()
    captured["graph"].replay()
    torch.cuda.synchronize()
    return captured["start"].elapsed_time(captured["end"]) * 1000  # ms -> us


def summarize(samples):
    return {"samples_us": samples, "median_us": statistics.median(samples),
            "min_us": min(samples), "max_us": max(samples)}


def run_case(reference, kernel, state, weights, case, mode, repeats, seed):
    original = make_input(reference, case["shape"], seed, weights)
    if mode == "check":
        static = clone_tree(original)
        output = kernel(static, state) if weights is not None else kernel(static)
        import torch
        torch.cuda.synchronize()
        return {"passed": True, "message": verify(reference, original, static, output, weights)}

    cap = prepare_graph(reference, kernel, state, weights, original)
    samples = []
    del original
    for repetition in range(repeats):
        original = make_input(reference, case["shape"], (seed + repetition + 1) % (2**63), weights)
        copy_tree(cap["static"], original)
        samples.append(replay_us(cap))
        message = verify(reference, original, cap["static"], cap["output"], weights)
        del original
    return {"passed": True, "message": message, **summarize(samples)}


def run_pair(reference, impls, weights, case, repeats, seed):
    """Candidate vs. reference implementation on the same fresh inputs, order alternating.

    impls is [(kernel, state) for candidate, (kernel, state) for against]. The speedup is the
    median of against/candidate ratios, each repetition pairing one replay of each side."""
    original = make_input(reference, case["shape"], seed, weights)
    caps = [prepare_graph(reference, kernel, state, weights, original) for kernel, state in impls]
    del original
    samples = ([], [])
    for repetition in range(repeats):
        original = make_input(reference, case["shape"], (seed + repetition + 1) % (2**63), weights)
        order = (0, 1) if repetition % 2 == 0 else (1, 0)
        for side in order:
            copy_tree(caps[side]["static"], original)
        for side in order:
            samples[side].append(replay_us(caps[side]))
        for cap in caps:
            message = verify(reference, original, cap["static"], cap["output"], weights)
        del original
    ratios = [b / a for a, b in zip(*samples)]
    return {"passed": True, "message": caps[0]["message"], **summarize(samples[0]),
            "against": summarize(samples[1]), "ratios": ratios,
            "speedup": statistics.median(ratios)}


def aggregate_cases(rows, key=lambda row: row["median_us"]):
    loads = {}
    for row in rows:
        loads[row["load"]] = loads.get(row["load"], 0.0) + key(row) * row["weight"] / 1000
    return {"loads_ms": loads, "geomean_ms": math.exp(statistics.mean(math.log(v) for v in loads.values())),
            "method": "sum(case median_us * weight) per load; geometric mean across loads"}


def run(options, report):
    import torch
    import triton

    script, cases = validate_options(ROOT, **options)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; use modal run modal_h200.py")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report["environment"] = {"gpu": torch.cuda.get_device_name(),
                             "torch": torch.__version__, "cuda": torch.version.cuda,
                             "triton": triton.__version__, "python": sys.version.split()[0]}
    if os.environ.get("MUSE_SGLANG_COMMIT"):
        from importlib.metadata import version
        report["environment"].update(
            sglang_commit=os.environ["MUSE_SGLANG_COMMIT"],
            image=os.environ["MUSE_SGLANG_IMAGE"],
            sglang_kernel=version("sglang-kernel"),
        )
    report["submission_sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
    print(json.dumps(report["environment"]), flush=True)
    sys.path.insert(0, str(script.parent))
    sys.path.insert(0, str(ROOT / options["problem"]))
    reference = load_module("reference", ROOT / options["problem"] / "reference.py")
    submission = load_module("submission", script)
    against = None
    if options.get("against"):
        if options["mode"] != "bench":
            raise ValueError("--against requires bench mode")
        against_path = (ROOT / options["problem"] / options["against"]).resolve()
        report["against_sha256"] = hashlib.sha256(against_path.read_bytes()).hexdigest()
        against = load_module("against_impl", against_path)
    with torch.no_grad():
        weights = (reference.generate_weights(reference.WEIGHTS_SEED)
                   if hasattr(reference, "generate_weights") else None)
        # Preserve the checker's weights even if prepare repacks its argument.
        state = submission.prepare(clone_tree(weights)) if weights is not None else None
        against_state = (against.prepare(clone_tree(weights)) if against is not None
                         and weights is not None else None)
        for index, case in cases:
            print(f"[{index}] {case['label']}", flush=True)
            row = {"case_id": index, **case}
            report["cases"].append(row)
            case_seed = (options["seed"] + index * 100003) % (2**63)
            try:
                if against is not None:
                    row.update(run_pair(reference, [(submission.kernel_fn, state),
                                                    (against.kernel_fn, against_state)],
                                        weights, case, options["repeats"], case_seed))
                else:
                    row.update(run_case(reference, submission.kernel_fn, state, weights, case,
                                        options["mode"], options["repeats"], case_seed))
            except Exception as exc:
                row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                raise
            line = "  PASS" + (f"  median {row['median_us']:.3f} us" if "median_us" in row else "")
            if "speedup" in row:
                line += f"  against {row['against']['median_us']:.3f} us  speedup {row['speedup']:.3f}x"
            print(line, flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    report["success"] = True
    if options["suite"] == "scored" and options["case_id"] == -1 and options["mode"] == "bench":
        report["aggregate"] = aggregate_cases(report["cases"])
        if against is not None:
            # Official-style: our time = against median / median paired speedup.
            report["aggregate_against"] = aggregate_cases(
                report["cases"], key=lambda row: row["against"]["median_us"])
            paired = aggregate_cases(
                report["cases"], key=lambda row: row["against"]["median_us"] / row["speedup"])
            report["aggregate_paired"] = paired
            report["speedup"] = report["aggregate_against"]["geomean_ms"] / paired["geomean_ms"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=PROBLEMS, default="muse_bf16_proj")
    parser.add_argument("--mode", choices=("check", "bench"), default="bench")
    parser.add_argument("--suite", choices=("smoke", "tests", "scored"), default="smoke")
    parser.add_argument("--submission", default="submission.py")
    parser.add_argument("--against", default="",
                        help="implementation in the problem folder to time alternately (e.g. baseline.py)")
    parser.add_argument("--case-id", type=int, default=-1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--result", type=Path)
    args = vars(parser.parse_args())
    list_cases, result_path = args.pop("list_cases"), args.pop("result")
    _, cases = validate_options(ROOT, **args)
    if list_cases:
        for index, case in cases:
            print(f"{index:2d}  {case['label']}")
        return 0
    report = {"success": False, "options": args, "cases": [],
              "timing": "CUDA graph device events; cold L2; fresh inputs; every replay checked",
              "official_score": False}
    try:
        run(args, report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    if result_path:
        result_path.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
