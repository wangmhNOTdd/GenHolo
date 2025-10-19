from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import torch  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 PyTorch，命令: pip install torch") from exc

try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("未检测到 RDKit，请在服务器执行: conda install -c conda-forge rdkit") from exc


HYBRIDIZATION_MAP: Dict[str, float] = {
    "SP": 1.0,
    "SP2": 2.0,
    "SP3": 3.0,
    "SP3D": 3.5,
    "SP3D2": 4.0,
}


@dataclass
class LigandEncoderConfig:
    projection_dim: int
    force_embed: bool = True


def _atom_base_features(atom: Chem.Atom) -> List[float]:
    hybrid = atom.GetHybridization().name
    features = [
        float(atom.GetAtomicNum()),
        float(atom.GetTotalDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs(includeNeighbors=True)),
        float(atom.GetMass()),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        HYBRIDIZATION_MAP.get(hybrid, 0.0),
    ]
    return features


class LigandFeatureBuilder:
    def __init__(self, cfg: LigandEncoderConfig) -> None:
        self.cfg = cfg

    def featurize(self, mol: Chem.Mol) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.cfg.force_embed and mol.GetNumConformers() == 0:
            temp = Chem.AddHs(mol)
            AllChem.EmbedMolecule(temp, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(temp)
            mol = Chem.RemoveHs(temp)
        conf = mol.GetConformer()
        atom_features: List[torch.Tensor] = []
        coords = []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])
            base = _atom_base_features(atom)
            charge = atom.GetDoubleProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0
            feat = torch.tensor(base + [charge], dtype=torch.float32)
            padded = torch.zeros(self.cfg.projection_dim, dtype=torch.float32)
            end = min(self.cfg.projection_dim, feat.numel())
            padded[:end] = feat[:end]
            atom_features.append(padded)
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        bond_index: List[Tuple[int, int]] = []
        bond_type: List[float] = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            order = float(bond.GetBondTypeAsDouble())
            bond_index.extend([(i, j), (j, i)])
            bond_type.extend([order, order])
        bond_index_tensor = torch.tensor(bond_index, dtype=torch.long) if bond_index else torch.empty(0, 2, dtype=torch.long)
        bond_type_tensor = torch.tensor(bond_type, dtype=torch.float32) if bond_type else torch.empty(0, dtype=torch.float32)

        angle_index: List[Tuple[int, int, int]] = []
        for atom in mol.GetAtoms():
            neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
            center = atom.GetIdx()
            for i in range(len(neighbors)):
                for j in range(len(neighbors)):
                    if i == j:
                        continue
                    angle_index.append((neighbors[i], center, neighbors[j]))
        angle_index_tensor = (
            torch.tensor(angle_index, dtype=torch.long) if angle_index else torch.empty(0, 3, dtype=torch.long)
        )

        torsion_index: List[Tuple[int, int, int, int]] = []
        for bond in mol.GetBonds():
            b = bond.GetBeginAtomIdx()
            c = bond.GetEndAtomIdx()
            nb_b = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(b).GetNeighbors() if nbr.GetIdx() != c]
            nb_c = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(c).GetNeighbors() if nbr.GetIdx() != b]
            for a in nb_b:
                for d in nb_c:
                    torsion_index.append((a, b, c, d))
                    torsion_index.append((d, c, b, a))
        torsion_index_tensor = (
            torch.tensor(torsion_index, dtype=torch.long) if torsion_index else torch.empty(0, 4, dtype=torch.long)
        )

        info = {
            "coords": coords_tensor,
            "atom_feats": torch.stack(atom_features) if atom_features else torch.empty(0, self.cfg.projection_dim),
            "bond_index": bond_index_tensor,
            "bond_type": bond_type_tensor,
            "angle_index": angle_index_tensor,
            "torsion_index": torsion_index_tensor,
        }
        return info["atom_feats"], info


def load_ligand_from_file(path: str) -> Chem.Mol:
    suffix = Path(path).suffix.lower()
    if suffix == ".sdf":
        suppl = Chem.SDMolSupplier(path, removeHs=False)
        mol = suppl[0]
    elif suffix in {".mol2", ".mol"}:
        mol = Chem.MolFromMol2File(path, removeHs=False)
    else:
        mol = Chem.MolFromPDBFile(path, removeHs=False)
    if mol is None:
        raise ValueError(f"无法从 {path} 解析出 RDKit 分子对象")
    mol = Chem.RemoveHs(mol)
    if mol.GetNumConformers() == 0:
        tmp = Chem.AddHs(mol)
        AllChem.EmbedMolecule(tmp, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(tmp)
        mol = Chem.RemoveHs(tmp)
    AllChem.ComputeGasteigerCharges(mol)
    return mol
