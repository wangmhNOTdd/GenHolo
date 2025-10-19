from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List

try:
    import torch  # type: ignore
    import torch.backends.cudnn as cudnn  # type: ignore
    import torch.nn.functional as F  # type: ignore
    from torch.cuda.amp import GradScaler, autocast  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 PyTorch，命令: pip install torch") from exc

from ..data.plinder_stage_a import StageADataset, collate_stage_a
from ..models.latent_decoder import DecoderConfig, LatentDecoder
from ..training.losses import (
    angle_loss,
    chemical_loss,
    contact_repel_loss,
    geometry_loss,
    huber_l1_loss,
    torsion_loss,
)
from ..utils.config import load_config
from ..utils.ema import ExponentialMovingAverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage A 解码器训练")
    parser.add_argument("--config", type=str, default="configs/stage_a.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--only-eval", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_dataloaders(cfg: Dict) -> Dict[str, DataLoader]:
    cache_root = Path(cfg["dataset"]["cache_root"])
    split_yaml = Path(cfg["dataset"]["split_yaml"])
    loaders: Dict[str, DataLoader] = {}
    for split in ["train", "val"]:
        dataset = StageADataset(cache_root, split_yaml, split)
        num_workers = cfg["training"].get("num_workers", 4)
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": cfg["training"]["micro_batch_size"],
            "shuffle": split == "train",
            "num_workers": num_workers,
            "pin_memory": cfg["training"].get("pin_memory", True),
            "collate_fn": collate_stage_a,
            "drop_last": False,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = cfg["training"].get("prefetch_factor", 2)
        loader = DataLoader(**loader_kwargs)
        loaders[split] = loader
    return loaders


def create_model(cfg: Dict, device: torch.device) -> LatentDecoder:
    model_cfg = DecoderConfig(**cfg["model"])
    model = LatentDecoder(model_cfg)
    model.to(device)
    return model


def compute_losses(
    pred: torch.Tensor,
    pred_mask: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    meta: Dict[str, List[torch.Tensor]],
    cfg: Dict,
) -> Dict[str, torch.Tensor]:
    delta = cfg["loss"].get("huber_delta", 1.0)
    lambda_repel = cfg["loss"].get("lambda_repel", 0.1)
    lambda_geom = cfg["loss"].get("lambda_geom", 0.5)
    lambda_chem = cfg["loss"].get("lambda_chem", 0.2)
    lambda_l1 = cfg["loss"].get("lambda_l1", 1.0)
    rc = cfg["loss"].get("contact_repel_rc", 2.0)
    beta = cfg["loss"].get("contact_repel_beta", 4.0)

    pred = pred[:, : target.size(1)]
    pred_mask = pred_mask[:, : target.size(1)]

    losses = {}
    losses["huber"] = huber_l1_loss(pred, target, valid_mask, delta)
    chem = chemical_loss(pred, target, meta)
    ang = angle_loss(pred, target, meta)
    tor = torsion_loss(pred, target, meta)
    losses["chem"] = chem + ang + tor
    losses["geom"] = geometry_loss(pred, target, valid_mask)
    losses["repel"] = contact_repel_loss(pred, meta, rc, beta)

    losses["total"] = (
        lambda_l1 * losses["huber"]
        + lambda_chem * losses["chem"]
        + lambda_geom * losses["geom"]
        + lambda_repel * losses["repel"]
    )
    return losses


def train_one_epoch(
    model: LatentDecoder,
    loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    ema: ExponentialMovingAverage,
    cfg: Dict,
    device: torch.device,
    epoch: int,
    global_step: int,
    autocast_kwargs: Dict,
) -> int:
    model.train()
    micro_batch = cfg["training"]["micro_batch_size"]
    accum = cfg["training"].get("grad_accum_steps", 1)
    tau = cfg["augmentation"].get("latent_noise_tau", 0.7)
    max_grad_norm = cfg["training"].get("max_grad_norm", 1.0)

    optimizer.zero_grad(set_to_none=True)
    train_loader = loaders["train"]
    steps_per_epoch = math.ceil(len(train_loader) / accum)

    for batch_idx, batch in enumerate(train_loader):
        latent = batch["latent"].to(device)
        mask = batch["mask"].to(device)
        segment_ids = batch["segment_ids"].to(device)
        ligand_coords = batch["ligand_coords"].to(device)
        ligand_valid = batch["ligand_valid"].to(device)

        sigma = torch.rand(latent.size(0), 1, 1, device=device) * tau
        noise = torch.randn_like(latent)
        latent_noisy = latent + sigma * noise

        with autocast(**autocast_kwargs):
            pred_coords, pred_mask = model(latent_noisy, mask, segment_ids, batch["meta"])
            loss_dict = compute_losses(pred_coords, pred_mask, ligand_coords, ligand_valid, batch["meta"], cfg)
            loss = loss_dict["total"] / accum

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            scheduler.step()
            global_step += 1

            if global_step % cfg["training"].get("log_interval", 50) == 0:
                lr = scheduler.get_last_lr()[0]
                detached = {k: v.detach().item() for k, v in loss_dict.items()}
                msg = (
                    f"epoch={epoch} step={global_step} lr={lr:.3e} "
                    f"loss={detached['total']:.4f} "
                    f"huber={detached['huber']:.4f} chem={detached['chem']:.4f} "
                    f"geom={detached['geom']:.4f} repel={detached['repel']:.4f}"
                )
                print(msg)

    return global_step


def evaluate(
    model: LatentDecoder,
    loader: DataLoader,
    cfg: Dict,
    device: torch.device,
    ema: ExponentialMovingAverage | None = None,
) -> Dict[str, float]:
    backup = None
    if ema is not None:
        backup = ema.apply_shadow(model)
    model.eval()
    total = {"total": 0.0, "huber": 0.0, "chem": 0.0, "geom": 0.0, "repel": 0.0}
    count = 0
    for batch in loader:
        latent = batch["latent"].to(device)
        mask = batch["mask"].to(device)
        segment_ids = batch["segment_ids"].to(device)
        ligand_coords = batch["ligand_coords"].to(device)
        ligand_valid = batch["ligand_valid"].to(device)

        pred_coords, pred_mask = model(latent, mask, segment_ids, batch["meta"])
        loss_dict = compute_losses(pred_coords, pred_mask, ligand_coords, ligand_valid, batch["meta"], cfg)
        for key in total:
            total[key] += float(loss_dict[key].item())
        count += 1
    if backup is not None and ema is not None:
        ema.restore(model, backup)
    if count == 0:
        return {k: 0.0 for k in total}
    return {k: v / count for k, v in total.items()}


def save_checkpoint(
    model: LatentDecoder,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    step: int,
    cfg: Dict,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": cfg,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    set_seed(cfg["training"].get("seed", 42))
    loaders = build_dataloaders(cfg)
    model = create_model(cfg, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        betas=tuple(cfg["training"].get("betas", (0.9, 0.95))),
        eps=cfg["training"].get("eps", 1e-8),
        weight_decay=cfg["training"].get("weight_decay", 0.05),
    )
    total_updates = cfg["training"]["epochs"] * math.ceil(len(loaders["train"]) / cfg["training"].get("grad_accum_steps", 1))

    def lr_lambda(step: int) -> float:
        warmup = cfg["training"].get("warmup_steps", 1000)
        if step < warmup:
            return max(step / max(1, warmup), 1e-4)
        progress = (step - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    amp_dtype = cfg["training"].get("amp_dtype", "bf16").lower()
    amp_enabled = device.type == "cuda" and amp_dtype in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == "fp16")
    ema = ExponentialMovingAverage(model, decay=cfg["training"].get("ema", {}).get("decay", 0.9995))

    start_epoch = 1
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint.get("epoch", 1)
        global_step = checkpoint.get("step", 0)

    if args.only_eval:
        metrics = evaluate(model, loaders["val"], cfg, device, ema)
        print("Validation:", metrics)
        return

    best_val = float("inf")
    output_dir = Path(cfg["paths"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        epoch_start = time.time()
        global_step = train_one_epoch(
            model,
            loaders,
            optimizer,
            scaler,
            scheduler,
            ema,
            cfg,
            device,
            epoch,
            global_step,
            {"enabled": amp_enabled, "dtype": autocast_dtype} if amp_enabled else {"enabled": False},
        )

        if cfg["training"].get("val_interval_steps", 2000) > 0:
            metrics = evaluate(model, loaders["val"], cfg, device, ema)
            val_loss = metrics["total"]
            elapsed = time.time() - epoch_start
            print(f"Epoch {epoch} validation loss={val_loss:.4f} (elapsed {elapsed/60:.1f} min)")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, optimizer, scaler, epoch, global_step, cfg, output_dir / "best.pt")

        if epoch % 1 == 0:
            save_checkpoint(model, optimizer, scaler, epoch, global_step, cfg, output_dir / f"epoch_{epoch}.pt")

    save_checkpoint(model, optimizer, scaler, cfg["training"]["epochs"], global_step, cfg, output_dir / "last.pt")


if __name__ == "__main__":
    main()
