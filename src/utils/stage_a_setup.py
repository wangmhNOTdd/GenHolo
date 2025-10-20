from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

try:
    import torch  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 PyTorch，命令: pip install torch") from exc

from ..data.cache_builder import DatasetConfig
from ..features.ligand import LigandEncoderConfig
from ..features.protein import ProteinEncoderConfig
from ..features.unimol_ligand import UniMolLigandEncoderConfig
from ..features.esm3_protein import ESM3ProteinEncoderConfig


def build_stage_a_configs(cfg: dict) -> Tuple[
    DatasetConfig,
    Union[ESM3ProteinEncoderConfig, ProteinEncoderConfig],
    Union[UniMolLigandEncoderConfig, LigandEncoderConfig]
]:
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

    # 根据配置选择蛋白质编码器类型
    protein_type = cfg["encoders"]["protein"].get("type", "esm")
    if protein_type == "esm3":
        protein_cfg = ESM3ProteinEncoderConfig(
            model_name=cfg["encoders"]["protein"]["model_name"],
            device=cfg["encoders"]["protein"].get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            fp16=cfg["encoders"]["protein"].get("fp16", False),
            chunk_size=cfg["encoders"]["protein"].get("chunk_size", 1024),
            proj_in=cfg["encoders"]["protein"]["projection"]["in_dim"],
            proj_out=cfg["encoders"]["protein"]["projection"]["out_dim"],
            proj_checkpoint=cfg["encoders"]["protein"]["projection"].get("checkpoint"),
        )
    else:
        # 传统ESM编码器
        protein_cfg = ProteinEncoderConfig(
            model_name=cfg["encoders"]["protein"]["model_name"],
            device=cfg["encoders"]["protein"].get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            fp16=cfg["encoders"]["protein"].get("fp16", False),
            chunk_size=cfg["encoders"]["protein"].get("chunk_size", 1024),
            proj_in=cfg["encoders"]["protein"]["projection"]["in_dim"],
            proj_out=cfg["encoders"]["protein"]["projection"]["out_dim"],
            proj_checkpoint=cfg["encoders"]["protein"]["projection"].get("checkpoint"),
        )

    # 根据配置选择配体编码器类型
    ligand_type = cfg["encoders"]["ligand"].get("type", "rdkit_gnn")
    if ligand_type == "unimol":
        ligand_cfg = UniMolLigandEncoderConfig(
            projection_dim=cfg["encoders"]["ligand"].get("projection_dim", 512),
            max_atoms=cfg["encoders"]["ligand"].get("max_atoms", 256),
            use_gpu=cfg["encoders"]["ligand"].get("use_gpu", True),
            remove_hs=cfg["encoders"]["ligand"].get("remove_hs", True),
        )
    else:
        # 传统RDKit编码器
        ligand_cfg = LigandEncoderConfig(
            projection_dim=cfg["encoders"]["ligand"].get("projection_dim", 512),
            force_embed=cfg["encoders"]["ligand"].get("force_rdkit_embed", True),
        )

    return dataset_cfg, protein_cfg, ligand_cfg
