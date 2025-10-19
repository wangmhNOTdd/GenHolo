from __future__ import annotations

import argparse

from src.data.cache_builder import StageACacheBuilder
from src.utils.config import load_config
from src.utils.stage_a_setup import build_stage_a_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 Stage A 训练所需的潜向量缓存")
    parser.add_argument("--config", type=str, default="configs/stage_a.yaml", help="配置文件路径")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=("train", "val", "test"),
        help="需要构建缓存的划分名称",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="每个划分最多处理的样本数，用于测试",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg, protein_cfg, ligand_cfg = build_stage_a_configs(cfg)

    builder = StageACacheBuilder(dataset_cfg, protein_cfg, ligand_cfg)
    for split in args.splits:
        print(f"开始构建 {split} 分割缓存...")
        builder.build_split(split, limit=args.limit)
        print(f"完成 {split}")


if __name__ == "__main__":
    main()
