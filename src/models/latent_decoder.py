from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 PyTorch，命令: pip install torch") from exc


@dataclass
class DecoderConfig:
    latent_dim: int
    hidden_dim: int
    num_layers: int
    edge_dim: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    heads: int | None = None
    coordinate_residual: bool = True
    use_edge_norm: bool = True


def _get_activation(name: str):
    name = name.lower()
    if name == "gelu":
        return F.gelu
    if name == "silu":
        return F.silu
    if name == "relu":
        return F.relu
    raise ValueError(f"不支持的激活函数: {name}")


class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.act = _get_activation(activation)
        self.edge_linear1 = nn.Linear(hidden_dim * 2 + edge_dim + 1, hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.edge_dropout = nn.Dropout(dropout)
        self.edge_linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.node_linear1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.node_dropout = nn.Dropout(dropout)
        self.node_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        update_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return h, x
        src, dst = edge_index
        diff = x[src] - x[dst]
        dist = torch.linalg.norm(diff, dim=-1, keepdim=True)
        edge_input = torch.cat([h[src], h[dst], dist, edge_attr], dim=-1)
        m_ij = self.edge_linear1(edge_input)
        m_ij = self.edge_norm(m_ij)
        m_ij = self.edge_dropout(self.act(m_ij))
        m_ij = self.edge_linear2(m_ij)

        coord_coef = self.coord_mlp(m_ij)
        coord_update = diff * coord_coef
        delta = torch.zeros_like(x)
        delta.index_add_(0, src, coord_update * update_mask[src].unsqueeze(-1))
        x = x + delta

        agg = torch.zeros_like(h)
        agg.index_add_(0, src, m_ij)
        node_input = torch.cat([h, agg], dim=-1)
        node_out = self.node_linear1(node_input)
        node_out = self.node_norm(node_out)
        node_out = self.node_dropout(self.act(node_out))
        node_out = self.node_linear2(node_out)
        h = h + node_out
        return h, x


class LatentDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.layers = nn.ModuleList(
            [EGNNLayer(cfg.hidden_dim, cfg.edge_dim, cfg.dropout, cfg.activation) for _ in range(cfg.num_layers)]
        )
        self.coord_init = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 3),
        )
        self.coord_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 3),
        )

    def forward(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
        segment_ids: torch.Tensor,
        meta: Dict[str, List[torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = latent.device
        batch_size = latent.size(0)
        pred_coords: List[torch.Tensor] = []
        for idx in range(batch_size):
            valid = mask[idx]
            tokens = latent[idx][valid]
            segments = segment_ids[idx][valid]
            protein_mask = segments == 0
            ligand_mask = segments == 1
            protein_tokens = tokens[protein_mask]
            ligand_tokens = tokens[ligand_mask]

            n_protein = protein_tokens.size(0)
            n_ligand = ligand_tokens.size(0)
            if n_ligand == 0:
                raise ValueError("批次中某个样本的配体 token 数为 0")

            node_feats = torch.cat([protein_tokens, ligand_tokens], dim=0)
            node_feats = self.input_proj(node_feats)

            pocket_coords = meta["pocket_coords"][idx].to(device)
            if pocket_coords.size(0) != n_protein:
                raise ValueError("口袋坐标数量与蛋白 token 数不一致")
            ligand_init = self.coord_init(node_feats[n_protein:])
            coords = torch.zeros(node_feats.size(0), 3, device=device)
            coords[:n_protein] = pocket_coords
            coords[n_protein:] = ligand_init

            update_mask = torch.zeros(node_feats.size(0), dtype=torch.bool, device=device)
            update_mask[n_protein:] = True

            edge_index, edge_attr = self._build_edges(meta, idx, n_protein, n_ligand, device)
            h = node_feats
            x = coords
            for layer in self.layers:
                h, x = layer(h, x, edge_index, edge_attr, update_mask)

            ligand_feat = h[n_protein:]
            ligand_coord = x[n_protein:]
            if self.cfg.coordinate_residual:
                ligand_coord = ligand_coord + self.coord_head(ligand_feat)
            else:
                ligand_coord = self.coord_head(ligand_feat)
            pred_coords.append(ligand_coord)

        max_atoms = max(item.size(0) for item in pred_coords)
        padded = latent.new_zeros(batch_size, max_atoms, 3)
        mask_out = latent.new_zeros(batch_size, max_atoms, dtype=torch.bool)
        for idx, coords in enumerate(pred_coords):
            num_atoms = coords.size(0)
            padded[idx, :num_atoms] = coords
            mask_out[idx, :num_atoms] = True
        return padded, mask_out

    def _build_edges(
        self,
        meta: Dict[str, List[torch.Tensor]],
        batch_idx: int,
        n_protein: int,
        n_ligand: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bond_index = meta["bond_index"][batch_idx].to(device)
        bond_type = (
            meta["bond_type"][batch_idx].to(device)
            if meta["bond_type"][batch_idx].numel()
            else torch.empty(0, device=device, dtype=torch.float32)
        )
        contact_index = meta["contact_index"][batch_idx].to(device)

        edges_i: List[int] = []
        edges_j: List[int] = []
        edge_attr: List[List[float]] = []

        for edge_idx, (u, v) in enumerate(bond_index.tolist()):
            ui = n_protein + u
            vj = n_protein + v
            edges_i.extend([ui, vj])
            edges_j.extend([vj, ui])
            weight = float(bond_type[edge_idx].item()) if bond_type.numel() else 1.0
            attr = [weight, 0.0]
            edge_attr.extend([attr, attr])

        for lig_idx, prot_idx in contact_index.tolist():
            li = n_protein + lig_idx
            pj = prot_idx
            if pj >= n_protein:
                continue
            edges_i.extend([li, pj, pj, li])
            edges_j.extend([pj, li, li, pj])
            attr_lp = [0.0, 1.0]
            edge_attr.extend([attr_lp, attr_lp, attr_lp, attr_lp])

        if not edges_i:
            return (
                torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0, self.cfg.edge_dim, device=device),
            )
        edge_index = torch.tensor([edges_i, edges_j], dtype=torch.long, device=device)
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32, device=device)
        return edge_index, edge_attr_tensor