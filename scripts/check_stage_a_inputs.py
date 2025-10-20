from __future__ import annotations

import argparse
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Tuple
from zipfile import ZipFile

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pandas as pd  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 pandas，命令: pip install pandas") from exc

from src.data.utils import load_split_ids
from src.utils.config import load_config
from src.utils.stage_a_setup import build_stage_a_configs

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from rdkit import Chem  # type: ignore
    RDKit_AVAILABLE = True
except ImportError:
    RDKit_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 Stage A 所需数据文件是否齐全")
    parser.add_argument("--config", type=str, default="configs/stage_a.yaml", help="配置文件路径")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=("train", "val", "test"),
        help="需要检查的划分",
    )
    parser.add_argument("--limit", type=int, default=5, help="每个划分最多抽样多少条记录")
    return parser.parse_args()


def load_annotation(table_path: Path, system_id_column: str) -> pd.DataFrame:
    if not table_path.exists():
        raise FileNotFoundError(f"annotation 表不存在: {table_path}")
    df = pd.read_parquet(table_path)
    if system_id_column not in df.columns:
        raise KeyError(f"annotation 表缺少列 {system_id_column}")
    return df.set_index(system_id_column, drop=False)


def _check_zip(zip_path: Path, system_id: str, pattern: str) -> Tuple[bool, str]:
    if not zip_path.exists():
        return False, f"zip 不存在: {zip_path}"
    with ZipFile(zip_path, "r") as zf:
        matches = [name for name in zf.namelist() if system_id in name and fnmatch(Path(name).name, pattern)]
    if matches:
        return True, matches[0]
    return False, f"zip 中未找到匹配 {pattern} 的文件"


def check_holo(dataset_cfg, record) -> Tuple[bool, str]:
    value = record.get(dataset_cfg.holo_structure_column, "")
    if isinstance(value, str) and value:
        path = dataset_cfg.root / value
        if path.exists() and path.is_file() and path.suffix != ".zip":
            return True, str(path)
        if path.suffix == ".zip" and path.exists():
            ok, detail = _check_zip(path, record[dataset_cfg.system_id_column], dataset_cfg.holo_selector)
            if ok:
                return True, f"zip:{detail}"
            return False, detail
    code = record.get(dataset_cfg.two_char_column)
    if isinstance(code, str) and code:
        zip_path = dataset_cfg.systems_dir / f"{code}.zip"
        ok, detail = _check_zip(zip_path, record[dataset_cfg.system_id_column], dataset_cfg.holo_selector)
        if ok:
            return True, f"zip:{detail}"
        return False, detail
    return False, "未提供 holo 结构路径或匹配失败"


def check_ligand(dataset_cfg, record) -> Tuple[bool, str]:
    value = record.get(dataset_cfg.ligand_file_column, "")
    if isinstance(value, str) and value:
        path = dataset_cfg.root / value
        if path.exists() and path.is_file() and path.suffix != ".zip":
            return True, str(path)
        if path.suffix == ".zip" and path.exists():
            ok, detail = _check_zip(path, record[dataset_cfg.system_id_column], dataset_cfg.ligand_selector)
            if ok:
                return True, f"zip:{detail}"
            return False, detail
    code = record.get(dataset_cfg.two_char_column)
    if isinstance(code, str) and code:
        # 尝试在linked_structures目录下查找配体文件
        zip_path = dataset_cfg.linked_struct_dir / f"{code[:2]}.zip"  # 使用前两位字符
        ok, detail = _check_zip(zip_path, record[dataset_cfg.system_id_column], dataset_cfg.ligand_selector)
        if ok:
            return True, f"zip:{detail}"
        else:
            # 如果在linked_structures中找不到，尝试在systems目录中查找
            system_zip_path = dataset_cfg.systems_dir / f"{code[:2]}.zip"
            ok, detail = _check_zip(system_zip_path, record[dataset_cfg.system_id_column], dataset_cfg.ligand_selector)
            if ok:
                return True, f"zip:{detail}"
        return False, detail
    return False, "未提供配体文件路径或匹配失败"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg, protein_cfg, ligand_cfg = build_stage_a_configs(cfg)

    print(f"使用蛋白质编码器类型: {type(protein_cfg).__name__}")
    print(f"使用配体编码器类型: {type(ligand_cfg).__name__}")
    print(f"蛋白质编码器输出维度: {protein_cfg.proj_out}")
    print(f"配体编码器输出维度: {ligand_cfg.projection_dim}")
    print()

    annotation = load_annotation(dataset_cfg.annotation_table, dataset_cfg.system_id_column)
    if dataset_cfg.two_char_column not in annotation.columns:
        raise KeyError(f"annotation 表缺少列 {dataset_cfg.two_char_column}")

    summary: Dict[str, Dict[str, List[str]]] = {}
    for split in args.splits:
        ids = load_split_ids(dataset_cfg.split_yaml, split)
        summary[split] = {"ok": [], "fail": []}
        for system_id in ids[: args.limit]:
            if system_id not in annotation.index:
                summary[split]["fail"].append(f"{system_id}: annotation 中不存在")
                continue
            record = annotation.loc[system_id]
            holo_ok, holo_detail = check_holo(dataset_cfg, record)
            lig_ok, lig_detail = check_ligand(dataset_cfg, record)
            if holo_ok and lig_ok:
                summary[split]["ok"].append(f"{system_id}: holo={holo_detail} ligand={lig_detail}")
            else:
                issues = []
                if not holo_ok:
                    issues.append(f"holo({holo_detail})")
                if not lig_ok:
                    issues.append(f"ligand({lig_detail})")
                summary[split]["fail"].append(f"{system_id}: {', '.join(issues)}")

    for split, result in summary.items():
        print(f"=== {split} ===")
        print(f"通过 {len(result['ok'])} 条，失败 {len(result['fail'])} 条")
        if result["fail"]:
            for item in result["fail"]:
                print(f"  [FAIL] {item}")
        if result["ok"]:
            for item in result["ok"]:
                print(f"  [OK] {item}")
        print()
    
    # 额外的依赖检查
    print("=== 依赖检查 ===")
    print(f"PyTorch 可用: {TORCH_AVAILABLE}")
    print(f"RDKit 可用: {RDKit_AVAILABLE}")
    print(f"蛋白质编码器类型: {type(protein_cfg).__name__}")
    print(f"配体编码器类型: {type(ligand_cfg).__name__}")


if __name__ == "__main__":
    main()
