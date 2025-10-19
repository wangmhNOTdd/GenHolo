from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 Stage A 训练")
    parser.add_argument("--config", type=str, default="configs/stage_a.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "src.training.train_stage_a",
        "--config",
        args.config,
        "--device",
        args.device,
    ]
    if args.resume:
        cmd.extend(["--resume", args.resume])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
