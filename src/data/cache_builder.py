from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
from zipfile import ZipFile

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 pandas，命令: pip install pandas") from exc

try:
    import torch  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 PyTorch，命令: pip install torch") from exc

try:
    from Bio.PDB import PDBParser  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 Biopython，命令: pip install biopython") from exc

from ..features.ligand import LigandEncoderConfig, LigandFeatureBuilder, load_ligand_from_file
from ..features.protein import ESMProteinEncoder, ProteinEncoderConfig
from ..features.unimol_ligand import UniMolLigandEncoder, UniMolLigandEncoderConfig
from ..features.esm3_protein import ESM3ProteinEncoder, ESM3ProteinEncoderConfig
from .utils import load_split_ids


@dataclass
class DatasetConfig:
    root: Path
    systems_dir: Path
    linked_struct_dir: Path
    cache_root: Path
    split_yaml: Path
    annotation_table: Path
    system_id_column: str
    two_char_column: str
    holo_structure_column: str
    ligand_file_column: str
    pocket_radius: float
    contact_cutoff: float
    max_residues: int
    max_tokens: int
    holo_selector: str
    ligand_selector: str
    ligand_fallback: str


@dataclass
class SelectedResidue:
    index: int
    centroid: torch.Tensor
    distance: float
    name: str


class StageACacheBuilder:
    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        protein_cfg: Union[ESM3ProteinEncoderConfig, ProteinEncoderConfig],
        ligand_cfg: Union[UniMolLigandEncoderConfig, LigandEncoderConfig],
    ) -> None:
        self.dataset_cfg = dataset_cfg
        # 根据配置类型创建相应的蛋白质编码器
        if isinstance(protein_cfg, ESM3ProteinEncoderConfig):
            self.protein_encoder = ESM3ProteinEncoder(protein_cfg)
        else:
            self.protein_encoder = ESMProteinEncoder(protein_cfg)
            
        # 根据配置类型创建相应的配体编码器
        if isinstance(ligand_cfg, UniMolLigandEncoderConfig):
            self.ligand_builder = UniMolLigandEncoder(ligand_cfg)
        else:
            self.ligand_builder = LigandFeatureBuilder(ligand_cfg)
            
        self.parser = PDBParser(QUIET=True)
        self.annotation = pd.read_parquet(dataset_cfg.annotation_table)
        if dataset_cfg.system_id_column not in self.annotation.columns:
            raise KeyError(f"annotation 表缺少列 {dataset_cfg.system_id_column}")
        self.annotation.set_index(dataset_cfg.system_id_column, inplace=True)
        os.makedirs(dataset_cfg.cache_root, exist_ok=True)
    
    def build_split(self, split: str, limit: Optional[int] = None) -> None:
        ids = load_split_ids(self.dataset_cfg.split_yaml, split)
        split_dir = self.dataset_cfg.cache_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        processed = 0
        for system_id in ids:
            cache_path = split_dir / f"{system_id}.pt"
            if cache_path.exists():
                continue
            record = self._fetch_record(system_id)
            try:
                cache = self._process_sample(system_id, record)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"构建样本 {system_id} 缓存失败") from exc
            torch.save(cache, cache_path)
            processed += 1
            if limit is not None and processed >= limit:
                break

    def _fetch_record(self, system_id: str) -> pd.Series:
        if system_id not in self.annotation.index:
            raise KeyError(f"annotation 表中不存在 system_id={system_id}")
        row = self.annotation.loc[[system_id]].iloc[0]
        return row

    def _process_sample(self, system_id: str, record: pd.Series) -> Dict[str, object]:
        holo_pdb = self._load_holo_pdb(system_id, record)
        ligand_mol = self._load_ligand(system_id, record)
        ligand_feats, ligand_info = self.ligand_builder.featurize(ligand_mol)
        ligand_feats = ligand_feats.to(torch.float32).cpu()
        ligand_coords = ligand_info["coords"].to(torch.float32).cpu()

        structure = self.parser.get_structure(system_id, io.StringIO(holo_pdb))
        residues = list(structure.get_residues())
        selected = self._select_residues(residues, ligand_coords)
        if not selected:
            raise RuntimeError(f"样本 {system_id} 裁剪后无有效口袋残基")

        ligand_atoms = ligand_feats.size(0)
        max_protein_tokens = max(0, min(len(selected), self.dataset_cfg.max_tokens - ligand_atoms))
        if max_protein_tokens <= 0:
            raise RuntimeError(f"样本 {system_id} 口袋裁剪后无法在 token 限制内保留残基")
        selected = selected[: max_protein_tokens]

        residue_names = [item.name for item in selected]
        if residue_names:
            protein_tokens = self.protein_encoder.encode(residue_names)
        else:
            protein_tokens = torch.empty(0, self.protein_encoder.cfg.proj_out, dtype=torch.float32)
        protein_tokens = protein_tokens[: len(selected)].to(torch.float32).cpu()
        protein_coords = torch.stack([item.centroid for item in selected], dim=0).to(torch.float32).cpu()

        total_tokens = protein_tokens.size(0) + ligand_atoms
        if total_tokens > self.dataset_cfg.max_tokens:
            raise RuntimeError(
                f"样本 {system_id} token 总数 {total_tokens} 超出上限 {self.dataset_cfg.max_tokens}")

        latent = torch.cat([protein_tokens, ligand_feats], dim=0)
        mask = torch.ones(total_tokens, dtype=torch.bool)
        segment_ids = torch.cat(
            [
                torch.zeros(protein_tokens.size(0), dtype=torch.long),
                torch.ones(ligand_feats.size(0), dtype=torch.long),
            ],
            dim=0,
        )

        contact_index = self._compute_contacts(protein_coords, ligand_coords)
        ligand_payload = {
            "coords": ligand_coords,
            "atom_feats": ligand_feats,
            "bond_index": ligand_info["bond_index"].long(),
            "bond_type": ligand_info["bond_type"].to(torch.float32),
            "angle_index": ligand_info["angle_index"].long(),
            "torsion_index": ligand_info["torsion_index"].long(),
            "contact_index": contact_index,
        }

        return {
            "system_id": system_id,
            "latent": latent,
            "mask": mask,
            "segment_ids": segment_ids,
            "ligand": ligand_payload,
            "protein": {"coords": protein_coords},
        }

    def _compute_contacts(self, protein_coords: torch.Tensor, ligand_coords: torch.Tensor) -> torch.Tensor:
        if protein_coords.numel() == 0 or ligand_coords.numel() == 0:
            return torch.empty(0, 2, dtype=torch.long)
        diff = ligand_coords[:, None, :] - protein_coords[None, :, :]
        dist = torch.linalg.norm(diff, dim=-1)
        mask = dist <= self.dataset_cfg.contact_cutoff
        pairs = mask.nonzero(as_tuple=False)
        return pairs

    def _select_residues(self, residues: Iterable, ligand_coords: torch.Tensor) -> List[SelectedResidue]:
        candidates: List[SelectedResidue] = []
        ligand_np = ligand_coords.numpy()
        for idx, residue in enumerate(residues):
            hetfield, _, _ = residue.id
            if hetfield.strip():
                continue
            heavy_coords = [atom.get_coord() for atom in residue.get_atoms() if atom.element != "H"]
            if not heavy_coords:
                continue
            residue_np = np.array(heavy_coords)
            dist = np.linalg.norm(ligand_np[None, :, :] - residue_np[:, None, :], axis=-1)
            min_dist = float(dist.min())
            if min_dist <= self.dataset_cfg.pocket_radius:
                centroid = torch.tensor(residue_np.mean(axis=0), dtype=torch.float32)
                name = residue.get_resname().upper()
                candidates.append(SelectedResidue(idx, centroid, min_dist, name))
        candidates.sort(key=lambda item: item.distance)
        return candidates[: self.dataset_cfg.max_residues]

    def _load_holo_pdb(self, system_id: str, record: pd.Series) -> str:
        if self.dataset_cfg.holo_structure_column in record:
            path_value = record[self.dataset_cfg.holo_structure_column]
            if isinstance(path_value, str) and path_value:
                path = self.dataset_cfg.root / path_value
                if path.exists():
                    print(f"DEBUG: Found holo structure at direct path: {path}")
                    return path.read_text(encoding="utf-8")
        
        # 从系统ID中提取两位字符代码（例如 "101m__1__1.A__1.C_1.D" -> "10"）
        code = system_id[:2]
        zip_path = self.dataset_cfg.systems_dir / f"{code}.zip"
        print(f"DEBUG: Trying holo structure from zip: {zip_path}")
        if zip_path.exists():
            data = self._extract_from_zip(zip_path, system_id, self.dataset_cfg.holo_selector)
            print(f"DEBUG: Successfully extracted holo structure from zip for {system_id}")
            return data.decode("utf-8")
            
        print(f"DEBUG: Failed to find holo structure for {system_id}")
        raise FileNotFoundError(f"未找到 {system_id} 的 holo 结构文件")

    def _load_ligand(self, system_id: str, record: pd.Series):
        # 尝试从记录中直接获取配体文件路径
        if self.dataset_cfg.ligand_file_column in record:
            rel = record[self.dataset_cfg.ligand_file_column]
            if isinstance(rel, str) and rel:
                path = self.dataset_cfg.root / rel
                if path.exists():
                    print(f"DEBUG: Found ligand at direct path: {path}")
                    return load_ligand_from_file(str(path))
        
        # 使用 entry_pdb_id 查找 CIF 文件
        pdb_id = record[self.dataset_cfg.two_char_column]
        cif_path = self.dataset_cfg.linked_struct_dir / f"{pdb_id}_A.cif"
        print(f"DEBUG: Trying ligand CIF file: {cif_path}")
        if cif_path.exists():
            print(f"DEBUG: Successfully found ligand CIF file for {system_id}")
            return load_ligand_from_file(str(cif_path))
            
        print(f"DEBUG: Failed to find ligand for {system_id}")
        raise FileNotFoundError(f"未找到 {system_id} 的配体文件")

    def _extract_from_zip(
        self,
        zip_path: Path,
        system_id: str,
        selector: str,
    ) -> bytes:
        with ZipFile(zip_path, "r") as zf:
            candidates = [
                name
                for name in zf.namelist()
                if system_id in name and fnmatch(Path(name).name, selector)
            ]
            if not candidates:
                raise FileNotFoundError(f"zip {zip_path} 中未找到 {system_id} 对应 {selector}")
            target = candidates[0]
            data = zf.read(target)
            return data

    def _write_temp_file(self, system_id: str, data: bytes) -> str:
        tmp_dir = self.dataset_cfg.cache_root / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".sdf" if self.dataset_cfg.ligand_selector.endswith(".sdf") else ".pdb"
        path = tmp_dir / f"{system_id}{suffix}"
        with open(path, "wb") as fh:
            fh.write(data)
        return str(path)
