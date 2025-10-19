from __future__ import annotations

from typing import Dict, List

try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 PyTorch，命令: pip install torch") from exc


def huber_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float) -> torch.Tensor:
    diff = pred - target
    abs_diff = diff.abs()
    loss = torch.where(abs_diff <= delta, 0.5 * diff.pow(2) / delta, abs_diff - 0.5 * delta)
    loss = loss.sum(dim=-1)
    if mask is not None:
        loss = loss * mask
        denom = mask.sum().clamp(min=1)
    else:
        denom = torch.tensor(loss.numel() // loss.size(-1), device=loss.device).clamp(min=1)
    return loss.sum() / denom


def _lengths(coords: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    vec = coords[pairs[:, 0]] - coords[pairs[:, 1]]
    return torch.linalg.norm(vec, dim=-1)


def chemical_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    meta: Dict[str, List[torch.Tensor]],
) -> torch.Tensor:
    total = pred.new_tensor(0.0)
    count = 0
    for idx, bonds in enumerate(meta["bond_index"]):
        if bonds.numel() == 0:
            continue
        unique = bonds[::2] if bonds.size(0) % 2 == 0 else bonds
        pred_l = _lengths(pred[idx], unique)
        true_l = _lengths(target[idx], unique)
        total = total + F.mse_loss(pred_l, true_l, reduction="sum")
        count += pred_l.numel()
    if count == 0:
        return total
    return total / count


def angle_loss(pred: torch.Tensor, target: torch.Tensor, meta: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
    total = pred.new_tensor(0.0)
    count = 0
    for idx, angles in enumerate(meta["angle_index"]):
        if angles.numel() == 0:
            continue
        triplets = angles[::2] if angles.size(0) % 2 == 0 else angles
        p = _angle_values(pred[idx], triplets)
        q = _angle_values(target[idx], triplets)
        total = total + F.mse_loss(p, q, reduction="sum")
        count += p.numel()
    if count == 0:
        return total
    return total / count


def torsion_loss(pred: torch.Tensor, target: torch.Tensor, meta: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
    total = pred.new_tensor(0.0)
    count = 0
    for idx, torsions in enumerate(meta["torsion_index"]):
        if torsions.numel() == 0:
            continue
        quads = torsions[::2] if torsions.size(0) % 2 == 0 else torsions
        p = _torsion_angles(pred[idx], quads)
        q = _torsion_angles(target[idx], quads)
        total = total + F.mse_loss(p, q, reduction="sum")
        count += p.numel()
    if count == 0:
        return total
    return total / count


def geometry_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    total = pred.new_tensor(0.0)
    count = 0
    for idx in range(pred.size(0)):
        mask = valid_mask[idx]
        n_atoms = mask.sum()
        if n_atoms < 2:
            continue
        coords_pred = pred[idx][mask]
        coords_true = target[idx][mask]
        dist_pred = torch.cdist(coords_pred, coords_pred, p=2)
        dist_true = torch.cdist(coords_true, coords_true, p=2)
        diff = (dist_pred - dist_true).abs()
        upper = torch.triu(diff, diagonal=1)
        total = total + upper.sum()
        count += coords_pred.size(0) * (coords_pred.size(0) - 1) / 2
    if count == 0:
        return total
    return total / count


def contact_repel_loss(
    pred: torch.Tensor,
    meta: Dict[str, List[torch.Tensor]],
    rc: float,
    beta: float,
) -> torch.Tensor:
    total = pred.new_tensor(0.0)
    count = 0
    for idx, contacts in enumerate(meta["contact_index"]):
        if contacts.numel() == 0:
            continue
        ligand_coords = pred[idx]
        pocket_coords = meta["pocket_coords"][idx].to(pred.device)
        lig = ligand_coords[contacts[:, 0]]
        poc = pocket_coords[contacts[:, 1]]
        dist = torch.linalg.norm(lig - poc, dim=-1)
        violation = torch.clamp(rc - dist, min=0.0)
        total = total + (violation.pow(2) * beta).sum()
        count += contacts.size(0)
    if count == 0:
        return total
    return total / count


def _angle_values(coords: torch.Tensor, triplets: torch.Tensor) -> torch.Tensor:
    a = coords[triplets[:, 0]]
    b = coords[triplets[:, 1]]
    c = coords[triplets[:, 2]]
    v1 = F.normalize(a - b, dim=-1, eps=1e-8)
    v2 = F.normalize(c - b, dim=-1, eps=1e-8)
    cos_theta = (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cos_theta)


def _torsion_angles(coords: torch.Tensor, quads: torch.Tensor) -> torch.Tensor:
    p0 = coords[quads[:, 0]]
    p1 = coords[quads[:, 1]]
    p2 = coords[quads[:, 2]]
    p3 = coords[quads[:, 3]]

    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = F.normalize(b1, dim=-1, eps=1e-8)
    v = b0 - (b0 * b1_norm).sum(dim=-1, keepdim=True) * b1_norm
    w = b2 - (b2 * b1_norm).sum(dim=-1, keepdim=True) * b1_norm
    v_norm = F.normalize(v, dim=-1, eps=1e-8)
    w_norm = F.normalize(w, dim=-1, eps=1e-8)
    x = (v_norm * w_norm).sum(dim=-1)
    y = torch.cross(b1_norm, v_norm, dim=-1)
    y = (y * w_norm).sum(dim=-1)
    return torch.atan2(y, x)
