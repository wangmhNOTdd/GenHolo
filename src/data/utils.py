from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 PyYAML，命令: pip install pyyaml") from exc


def load_split_ids(split_yaml: Path, split: str) -> List[str]:
    with open(split_yaml, "r", encoding="utf-8") as fh:
        data: Dict[str, Iterable[str]] = yaml.safe_load(fh)
    candidates = [split, split.upper(), split.capitalize(), f"{split}_ids"]
    for key in candidates:
        if key in data:
            items = data[key]
            if isinstance(items, dict) and "ids" in items:
                items = items["ids"]
            return [str(x) for x in items]
    raise KeyError(f"split.yaml 中未找到分割 '{split}'，可用键: {list(data.keys())}")


def list_cache_files(cache_root: Path, split: str, sample_ids: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    split_dir = cache_root / split
    for sample_id in sample_ids:
        expected = split_dir / f"{sample_id}.pt"
        if not expected.exists():
            raise FileNotFoundError(f"未找到缓存文件: {expected}")
        paths.append(expected)
    return paths
