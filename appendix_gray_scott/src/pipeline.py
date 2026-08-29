#!/usr/bin/env python3
"""Configuration-driven end-to-end Gray-Scott workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def run(stage: str, command: list[str], manifest: list[dict[str, object]]) -> None:
    print(f"\n[{len(manifest) + 1}/8] {stage}", flush=True)
    print("COMMAND:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    manifest.append({"stage": stage, "command": command})


def values(section: dict, names: list[str]) -> list[str]:
    output: list[str] = []
    for name in names:
        output.extend([f"--{name.replace('_', '-')}", str(section[name])])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sim, dataset = config["simulation"], config["dataset"]
    bench, sweep = config["benchmark"], config["parameter_sweep"]
    src, py = args.case_dir / "src", sys.executable
    args.work_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    sim_names = ["nx", "ny", "steps", "save_every", "dt", "du", "dv", "feed", "kill", "noise"]

    run("优化前后端数值一致性测试", [py, str(src / "test_case.py"), "--report", str(args.work_dir / "backend_consistency.json")], manifest)
    run("前处理：生成网格、配置快照和初值图", [py, str(src / "preprocess.py"), "--out-dir", str(args.work_dir), *values(sim, sim_names), "--samples", str(dataset["samples"]), "--seed", str(dataset["seed"])], manifest)
    run("生成计算流程、模板和并行模型图", [py, str(src / "make_diagrams.py"), "--out-dir", str(args.work_dir / "figures")], manifest)
    run("Numba + 样本级线程并行生成 HDF5", [py, str(src / "run_case.py"), "--out", str(args.work_dir / "grayscott_dataset.h5"), "--backend", str(dataset["backend"]), "--executor", str(dataset["executor"]), "--samples", str(dataset["samples"]), "--workers", str(dataset["workers"]), "--seed", str(dataset["seed"]), *values(sim, sim_names)], manifest)
    run("性能基准：Python/NumPy/Numba 和 1/2/4 worker", [py, str(src / "benchmark.py"), "--out-dir", str(args.work_dir / "benchmark"), *values(bench, ["nx", "ny", "steps", "save_every", "samples", "seed", "repeats"]), "--workers", *[str(item) for item in bench["workers"]]], manifest)
    run("物理后处理：场、方程项、频谱和 GIF", [py, str(src / "visualize.py"), "--input", str(args.work_dir / "grayscott_dataset.h5"), "--out-dir", str(args.work_dir / "results"), "--sample", "0000"], manifest)
    run("参数扫描：F-k 模式相图", [py, str(src / "parameter_sweep.py"), "--out-dir", str(args.work_dir / "parameter_sweep"), *values(sweep, ["nx", "ny", "steps", "save_every", "seed"]), "--dt", str(sim["dt"]), "--du", str(sim["du"]), "--dv", str(sim["dv"]), "--noise", str(sim["noise"]), "--feeds", *[str(item) for item in sweep["feeds"]], "--kills", *[str(item) for item in sweep["kills"]]], manifest)
    (args.work_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    run("自动验收全部数据、性能指标和图像", [py, str(src / "verify_results.py"), str(args.work_dir)], manifest)
    (args.work_dir / "command_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
