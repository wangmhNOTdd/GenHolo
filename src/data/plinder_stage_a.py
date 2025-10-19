from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

try:
    import torch  # type: ignore
    from torch.utils.data import Dataset  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 PyTorch，命令: pip install torch") from exc

from .utils import list_cache_files, load_split_ids


@dataclass
class StageASample:
    system_id: str
    latent: torch.Tensor
    mask: torch.Tensor
    segment_ids: torch.Tensor
    ligand_coords: torch.Tensor
    ligand_atom_feats: torch.Tensor
    ligand_bond_index: torch.Tensor
    ligand_bond_type: torch.Tensor
    ligand_angle_index: torch.Tensor
    ligand_torsion_index: torch.Tensor
    pocket_coords: torch.Tensor
    contact_index: torch.Tensor


class StageADataset(Dataset):
    def __init__(
        self,
        cache_root: Path,
        split_yaml: Path,
        split: str,
    ) -> None:
        sample_ids = load_split_ids(split_yaml, split)
        self.paths = list_cache_files(cache_root, split, sample_ids)

    def __len__(self) -> int:  # noqa: D401
        return len(self.paths)

    def __getitem__(self, index: int) -> StageASample:  # noqa: D401
        cache = torch.load(self.paths[index], map_location="cpu")
        ligand = cache["ligand"]
        protein = cache.get("protein", {})
        return StageASample(
            system_id=cache["system_id"],
            latent=cache["latent"].float(),
            mask=cache["mask"].bool(),
            segment_ids=cache["segment_ids"].long(),
            ligand_coords=ligand["coords"].float(),
            ligand_atom_feats=ligand["atom_feats"].float(),
            ligand_bond_index=ligand["bond_index"].long(),
            ligand_bond_type=ligand["bond_type"].float(),
            ligand_angle_index=ligand["angle_index"].long(),
            ligand_torsion_index=ligand["torsion_index"].long(),
            pocket_coords=protein.get("coords", torch.empty(0, 3)).float(),
            contact_index=ligand.get("contact_index", torch.empty(0, 2, dtype=torch.long)),
        )


def collate_stage_a(batch: List[StageASample]) -> Dict[str, torch.Tensor]:
    max_tokens = max(item.latent.size(0) for item in batch)
    max_ligand_atoms = max(item.ligand_coords.size(0) for item in batch)

    latent_batch = torch.zeros(len(batch), max_tokens, batch[0].latent.size(-1))
    mask_batch = torch.zeros(len(batch), max_tokens, dtype=torch.bool)
    segment_batch = torch.zeros(len(batch), max_tokens, dtype=torch.long)

    ligand_coords = torch.zeros(len(batch), max_ligand_atoms, 3)
    ligand_feats = torch.zeros(len(batch), max_ligand_atoms, batch[0].ligand_atom_feats.size(-1))

    ligand_valid = torch.zeros(len(batch), max_ligand_atoms, dtype=torch.bool)

    sample_meta: Dict[str, List] = {
        "bond_index": [],
        "bond_type": [],
        "angle_index": [],
        "torsion_index": [],
        "contact_index": [],
        "pocket_coords": [],
        "system_ids": [],
    }

    for idx, item in enumerate(batch):
        n_tokens = item.latent.size(0)
        n_ligand = item.ligand_coords.size(0)
        latent_batch[idx, :n_tokens] = item.latent
        mask_batch[idx, :n_tokens] = item.mask
        segment_batch[idx, :n_tokens] = item.segment_ids

        ligand_coords[idx, :n_ligand] = item.ligand_coords
        ligand_feats[idx, :n_ligand] = item.ligand_atom_feats
        ligand_valid[idx, :n_ligand] = True

        sample_meta["bond_index"].append(item.ligand_bond_index)
        sample_meta["bond_type"].append(item.ligand_bond_type)
        sample_meta["angle_index"].append(item.ligand_angle_index)
        sample_meta["torsion_index"].append(item.ligand_torsion_index)
        sample_meta["contact_index"].append(item.contact_index)
        sample_meta["pocket_coords"].append(item.pocket_coords)
        sample_meta["system_ids"].append(item.system_id)

    return {
        "latent": latent_batch,
        "mask": mask_batch,
        "segment_ids": segment_batch,
        "ligand_coords": ligand_coords,
        "ligand_feats": ligand_feats,
        "ligand_valid": ligand_valid,
        "meta": sample_meta,
    }
