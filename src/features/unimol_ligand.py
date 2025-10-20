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

try:
    import unimol  # type: ignore
    from unimol_tools import UniMolRepr  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "未检测到 UniMol，请先在服务器上安装 (pip install unimol unimol-tools)。"
    ) from exc


@dataclass
class UniMolLigandEncoderConfig:
    projection_dim: int
    max_atoms: int = 256
    use_gpu: bool = True
    remove_hs: bool = True
    featurize_method: str = "unimol"  # unimol or rdkit_gnn


class UniMolLigandEncoder:
    def __init__(self, cfg: UniMolLigandEncoderConfig) -> None:
        self.cfg = cfg
        # 初始化UniMol表示提取器
        self.unimol_repr = UniMolRepr(
            data_type='molecule',
            remove_hs=cfg.remove_hs,
            device='cuda' if cfg.use_gpu and torch.cuda.is_available() else 'cpu'
        )
        
        # 创建投影层
        self.projection = torch.nn.Linear(512, cfg.projection_dim)  # UniMol输出通常是512维
        self.projection.eval()

    def featurize(self, mol: Chem.Mol) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        使用UniMol对分子进行特征化
        
        Args:
            mol: RDKit分子对象
            
        Returns:
            (atom_features, info_dict) 其中atom_features是投影后的原子特征张量
        """
        # 使用UniMol获取分子表示
        smiles = Chem.MolToSmiles(mol)
        
        try:
            # 通过UniMol获取分子表示
            repr_data = self.unimol_repr.get_repr([smiles])
            # UniMol返回的表示通常是[分子数, 原子数, 特征维数]
            mol_repr = repr_data['repr'][0]  # 取第一个分子的表示
            
            # 转换为张量
            mol_repr_tensor = torch.tensor(mol_repr, dtype=torch.float32)
            
            # 获取分子的坐标信息
            if mol.GetNumConformers() > 0:
                conf = mol.GetConformer()
                coords = []
                for i in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    coords.append([pos.x, pos.y, pos.z])
                coords_tensor = torch.tensor(coords, dtype=torch.float32)
            else:
                # 如果没有坐标，生成随机坐标
                coords_tensor = torch.randn(mol_repr_tensor.size(0), 3)
            
            # 投影到目标维度
            projected_features = self.projection(mol_repr_tensor)
            
            # 生成分子图结构信息
            bond_index, bond_type = self._get_bond_info(mol)
            angle_index = self._get_angle_info(mol)
            torsion_index = self._get_torsion_info(mol)
            
            info = {
                "coords": coords_tensor,
                "atom_feats": projected_features,
                "bond_index": bond_index,
                "bond_type": bond_type,
                "angle_index": angle_index,
                "torsion_index": torsion_index,
            }
            
            return projected_features, info
        except Exception as e:
            # 如果UniMol失败，回退到传统的RDKit方法
            print(f"UniMol编码失败，使用回退方法: {e}")
            return self._fallback_encode(mol)

    def encode(self, mol: Chem.Mol) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        使用UniMol对分子进行编码（向后兼容方法）
        
        Args:
            mol: RDKit分子对象
            
        Returns:
            (atom_features, info_dict) 其中atom_features是投影后的原子特征张量
        """
        return self.featurize(mol)

    def _get_bond_info(self, mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取分子的键信息"""
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
        
        return bond_index_tensor, bond_type_tensor

    def _get_angle_info(self, mol: Chem.Mol) -> torch.Tensor:
        """获取分子的键角信息"""
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
        
        return angle_index_tensor

    def _get_torsion_info(self, mol: Chem.Mol) -> torch.Tensor:
        """获取分子的二面角信息"""
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
        
        return torsion_index_tensor

    def _fallback_encode(self, mol: Chem.Mol) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """回退编码方法，使用传统的RDKit特征"""
        # 生成分子坐标（如果不存在）
        if mol.GetNumConformers() == 0:
            temp = Chem.AddHs(mol)
            AllChem.EmbedMolecule(temp, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(temp)
            mol = Chem.RemoveHs(temp)
        
        conf = mol.GetConformer()
        coords = []
        atom_features = []
        
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])
            
            # 基本原子特征
            features = [
                float(atom.GetAtomicNum()),
                float(atom.GetTotalDegree()),
                float(atom.GetFormalCharge()),
                float(atom.GetTotalNumHs(includeNeighbors=True)),
                float(atom.GetMass()),
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
            ]
            
            # 添加杂化类型
            hybrid = atom.GetHybridization().name
            hybrid_map = {
                "SP": 1.0,
                "SP2": 2.0,
                "SP3": 3.0,
                "SP3D": 3.5,
                "SP3D2": 4.0,
            }
            features.append(hybrid_map.get(hybrid, 0.0))
            
            # 添加Gasteiger电荷
            charge = atom.GetDoubleProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0
            features.append(charge)
            
            feat_tensor = torch.tensor(features, dtype=torch.float32)
            padded = torch.zeros(self.cfg.projection_dim, dtype=torch.float32)
            end = min(self.cfg.projection_dim, feat_tensor.numel())
            padded[:end] = feat_tensor[:end]
            atom_features.append(padded)
        
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        atom_features_tensor = torch.stack(atom_features) if atom_features else torch.empty(0, self.cfg.projection_dim)
        
        # 生成分子图结构信息
        bond_index, bond_type = self._get_bond_info(mol)
        angle_index = self._get_angle_info(mol)
        torsion_index = self._get_torsion_info(mol)
        
        info = {
            "coords": coords_tensor,
            "atom_feats": atom_features_tensor,
            "bond_index": bond_index,
            "bond_type": bond_type,
            "angle_index": angle_index,
            "torsion_index": torsion_index,
        }
        
        return atom_features_tensor, info


def load_ligand_from_file(path: str) -> Chem.Mol:
    """从文件加载配体分子"""
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