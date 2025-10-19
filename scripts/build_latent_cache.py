from __future__ import annotations

import argparse
from pathlib import Path

try:
    import torch  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 PyTorch，命令: pip install torch") from exc

from src.data.cache_builder import DatasetConfig, StageACacheBuilder
from src.features.ligand import LigandEncoderConfig
from src.features.protein import ProteinEncoderConfig
from src.utils.config import load_config


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


def build_configs(cfg: dict) -> tuple[DatasetConfig, ProteinEncoderConfig, LigandEncoderConfig]:
    dataset_cfg = DatasetConfig(
        root=Path(cfg["dataset"]["root"]),
        systems_dir=Path(cfg["dataset"]["systems_dir"]),
        linked_struct_dir=Path(cfg["dataset"]["linked_structures_dir"]),
        cache_root=Path(cfg["dataset"]["cache_root"]),
        split_yaml=Path(cfg["dataset"]["split_yaml"]),
        annotation_table=Path(cfg["dataset"]["annotation_table"]),
        system_id_column=cfg["dataset"]["system_id_column"],
        two_char_column=cfg["dataset"]["two_char_column"],
        holo_structure_column=cfg["dataset"]["holo_structure_column"],
        ligand_file_column=cfg["dataset"]["ligand_file_column"],
        pocket_radius=cfg["dataset"].get("pocket_radius", 8.0),
        contact_cutoff=cfg["dataset"].get("contact_cutoff", 6.0),
        max_residues=cfg["dataset"].get("max_residues", 192),
        max_tokens=cfg["dataset"].get("max_tokens", 256),
        holo_selector=cfg["dataset"].get("holo_pdb_selector", "*_holo.pdb"),
        ligand_selector=cfg["dataset"].get("ligand_file_selector", "*_ligand.sdf"),
        ligand_fallback=cfg["dataset"].get("ligand_fallback_format", "pdb"),
    )

    enc_cfg = cfg["encoders"]["protein"]
    protein_cfg = ProteinEncoderConfig(
        model_name=enc_cfg["model_name"],
        device=enc_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        fp16=enc_cfg.get("fp16", False),
        chunk_size=enc_cfg.get("chunk_size", 1024),
        proj_in=enc_cfg["projection"]["in_dim"],
        proj_out=enc_cfg["projection"]["out_dim"],
        proj_checkpoint=enc_cfg["projection"].get("checkpoint"),
    )

    lig_cfg = cfg["encoders"]["ligand"]
    ligand_cfg = LigandEncoderConfig(
        projection_dim=lig_cfg.get("projection_dim", 512),
        force_embed=lig_cfg.get("force_rdkit_embed", True),
    )

    return dataset_cfg, protein_cfg, ligand_cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg, protein_cfg, ligand_cfg = build_configs(cfg)

    builder = StageACacheBuilder(dataset_cfg, protein_cfg, ligand_cfg)
    for split in args.splits:
        print(f"开始构建 {split} 分割缓存...")
        builder.build_split(split, limit=args.limit)
        print(f"完成 {split}")


if __name__ == "__main__":
    main()
