from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import torch  # type: ignore
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 PyTorch，命令: pip install torch") from exc

try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("未检测到 RDKit，请在服务器执行: conda install -c conda-forge rdkit") from exc


@dataclass
class PaiNNLigandEncoderConfig:
    projection_dim: int = 512
    hidden_channels: int = 256
    num_layers: int = 4
    num_rbf: int = 32
    cutoff: float = 6.0
    max_atoms: int = 256
    use_gpu: bool = True


class PaiNNLigandEncoder:
    def __init__(self, cfg: PaiNNLigandEncoderConfig) -> None:
        self.cfg = cfg
        
        # 尝试导入TorchMDNet的PaiNN
        try:
            from torchmdnet.models.painn import PaiNN
            self.painn = PaiNN(
                hidden_channels=cfg.hidden_channels,
                num_layers=cfg.num_layers,
                num_rbf=cfg.num_rbf,
                cutoff=cfg.cutoff
            )
        except ImportError:
            # 如果没有torchmdnet，创建一个简单的PaiNN替代实现
            self.painn = self._create_simple_painn()
        
        self.painn.eval()
        for p in self.painn.parameters():
            p.requires_grad = False  # 冻结
        
        # 创建投影层，将hidden_channels投影到目标维度
        self.projection = nn.Linear(cfg.hidden_channels, cfg.projection_dim, bias=False)
        self.projection.eval()

    def _create_simple_painn(self):
        """创建一个简单的PaiNN替代实现，用于测试目的"""
        # 这是一个简化的表示，实际的PaiNN实现会更复杂
        # 这里我们创建一个简单的图神经网络作为替代
        import torch.nn.functional as F
        
        class SimplePaiNN(nn.Module):
            def __init__(self, hidden_channels=256, num_layers=4, num_rbf=32, cutoff=6.0):
                super().__init__()
                self.hidden_channels = hidden_channels
                self.num_layers = num_layers
                self.cutoff = cutoff
                
                # 原子嵌入层
                self.atom_embedding = nn.Embedding(100, hidden_channels)  # 假设最多100种原子
                
                # 简单的消息传递层
                self.message_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_channels * 2, hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, hidden_channels)
                    ) for _ in range(num_layers)
                ])
                
            def forward(self, z, pos, batch, edge_index):
                # 原子嵌入
                h = self.atom_embedding(z)
                
                # 简单的消息传递
                for layer in self.message_layers:
                    # 获取边的特征
                    row, col = edge_index
                    msg_input = torch.cat([h[row], h[col]], dim=-1)
                    msg = layer(msg_input)
                    
                    # 聚合消息
                    agg_msg = torch.zeros_like(h)
                    agg_msg.index_add_(0, row, msg)
                    
                    # 更新节点特征
                    h = h + agg_msg
                
                return h
        
        return SimplePaiNN(
            hidden_channels=self.cfg.hidden_channels,
            num_layers=self.cfg.num_layers,
            num_rbf=self.cfg.num_rbf,
            cutoff=self.cfg.cutoff
        )

    def featurize(self, mol: Chem.Mol) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        使用PaiNN对分子进行特征化
        
        Args:
            mol: RDKit分子对象
            
        Returns:
            (atom_features, info_dict) 其中atom_features是投影后的原子特征张量
        """
        try:
            # 获取原子类型和坐标
            N = mol.GetNumAtoms()
            Z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)  # [N]
            
            # 获取3D坐标
            if mol.GetNumConformers() == 0:
                # 如果没有3D构象，生成一个
                mol_copy = Chem.Mol(mol)
                Chem.AllChem.EmbedMolecule(mol_copy, Chem.AllChem.ETKDGv3())
                Chem.AllChem.UFFOptimizeMolecule(mol_copy, maxIters=200)
                conf = mol_copy.GetConformer()
            else:
                conf = mol.GetConformer()
            
            pos = torch.tensor([
                [conf.GetAtomPosition(i).x,
                 conf.GetAtomPosition(i).y,
                 conf.GetAtomPosition(i).z] for i in range(N)
            ], dtype=torch.float)  # [N,3]
            
            # 构建分子图：键 + 半径近邻
            edges = []
            for b in mol.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                edges += [(i, j), (j, i)]
            edge_index_bond = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E_bond]

            # 半径近邻
            cutoff = self.cfg.cutoff
            coords = pos.numpy()
            nbrs = []
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    if torch.norm(pos[i] - pos[j]) <= cutoff:
                        nbrs.append((i, j))
            edge_index_rad = torch.tensor(nbrs, dtype=torch.long).t().contiguous()  # [2, E_rad]

            # 合并边
            if edge_index_bond.numel() > 0 and edge_index_rad.numel() > 0:
                edge_index = torch.unique(torch.cat([edge_index_bond, edge_index_rad], dim=1), dim=1)
            elif edge_index_bond.numel() > 0:
                edge_index = edge_index_bond
            elif edge_index_rad.numel() > 0:
                edge_index = edge_index_rad
            else:
                # 如果没有边，创建自环以避免错误
                edge_index = torch.stack([torch.arange(N), torch.arange(N)], dim=0)

            # 创建批次张量（单分子：全0）
            batch = torch.zeros(N, dtype=torch.long)
            
            # 使用PaiNN编码
            with torch.no_grad():
                device = next(self.painn.parameters()).device if next(self.painn.parameters()).is_cuda else 'cpu'
                Z = Z.to(device)
                pos = pos.to(device)
                edge_index = edge_index.to(device)
                batch = batch.to(device)
                
                H = self.painn(z=Z, pos=pos, batch=batch, edge_index=edge_index)  # [N, hidden]
                
                # 投影到目标维度
                H_L = self.projection(H)  # [N, 512] - 这是per-atom tokens（配体分支）
            
            # 生成分子图结构信息
            bond_index, bond_type = self._get_bond_info(mol)
            angle_index = self._get_angle_info(mol)
            torsion_index = self._get_torsion_info(mol)
            
            info = {
                "coords": pos,
                "atom_feats": H_L,
                "bond_index": bond_index,
                "bond_type": bond_type,
                "angle_index": angle_index,
                "torsion_index": torsion_index,
            }
            
            return H_L, info
        except Exception as e:
            # 如果PaiNN失败，回退到传统的RDKit方法
            print(f"PaiNN编码失败，使用回退方法: {e}")
            return self._fallback_encode(mol)

    def encode(self, mol: Chem.Mol) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        使用PaiNN对分子进行编码（向后兼容方法）
        
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