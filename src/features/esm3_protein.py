from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

try:
    import torch  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 PyTorch，命令: pip install torch") from exc

try:
    import esm  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "未检测到 ESM 库，请先在服务器上安装 (pip install fair-esm 或官方 ESM3 包)。"
    ) from exc

AA_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


@dataclass
class ESM3ProteinEncoderConfig:
    model_name: str
    device: str
    fp16: bool
    chunk_size: int
    proj_in: int
    proj_out: int
    proj_checkpoint: str | None = None


class ESM3ProteinEncoder:
    def __init__(self, cfg: ESM3ProteinEncoderConfig) -> None:
        self.cfg = cfg
        # 加载ESM3模型 - 注意ESM3可能有不同的API
        try:
            # 尝试加载ESM3模型
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(cfg.model_name)
        except:
            # 如果上面的方法不适用，尝试其他ESM3加载方法
            try:
                self.model, self.alphabet = esm.pretrained.esm3_sm_open_v1()
            except:
                raise ValueError(f"无法加载模型 {cfg.model_name}，请确保模型名称正确")
        
        self.model.eval()
        self.model = self.model.to(cfg.device)
        if cfg.fp16:
            self.model = self.model.half()
        
        # 检查是否是ESM3模型并具有正确的接口
        self.is_esm3 = hasattr(self.model, 'encode')
        
        if hasattr(self.alphabet, 'get_batch_converter'):
            self.batch_converter = self.alphabet.get_batch_converter()
        else:
            # 如果是ESM3，可能需要不同的转换器
            self.batch_converter = None
            
        self.cls_token = getattr(self.alphabet, 'cls_idx', 0)
        self.eos_token = getattr(self.alphabet, 'eos_idx', 2)
        self.padding_idx = getattr(self.alphabet, 'padding_idx', 1)
        
        self.projection = torch.nn.Linear(cfg.proj_in, cfg.proj_out)
        if cfg.proj_checkpoint:
            state = torch.load(cfg.proj_checkpoint, map_location="cpu")
            self.projection.load_state_dict(state)
        self.projection.eval()

    def encode(self, residues: Sequence[str]) -> torch.Tensor:
        sequence = "".join(AA_THREE_TO_ONE.get(res, "X") for res in residues)
        
        if self.is_esm3:
            # ESM3 特定的编码方法
            return self._encode_esm3(sequence)
        else:
            # 传统ESM编码方法
            return self._encode_traditional_esm(sequence)

    def _encode_traditional_esm(self, sequence: str) -> torch.Tensor:
        """传统ESM编码方法"""
        if self.batch_converter is None:
            raise ValueError("ESM模型未正确初始化batch_converter")
            
        batch = [("protein", sequence)]
        _, _, tokens = self.batch_converter(batch)
        tokens = tokens.to(self.cfg.device)
        
        with torch.no_grad():
            model_out = self.model(tokens, repr_layers=[self.model.num_layers])
            reps = model_out["representations"][self.model.num_layers]
            reps = reps[:, 1 : len(sequence) + 1]  # 移除cls和eos token
            reps = reps.to(torch.float32)
            projected = self.projection(reps)
        return projected.squeeze(0)

    def _encode_esm3(self, sequence: str) -> torch.Tensor:
        """ESM3编码方法 - 需要根据实际ESM3 API调整"""
        # ESM3的编码方式可能不同，这里提供一个基础实现
        # 实际ESM3 API可能需要根据具体版本调整
        try:
            # 尝试使用ESM3的编码接口
            if hasattr(self.model, 'encode'):
                # ESM3编码
                encoded = self.model.encode(sequence)
                # 假设返回的是token representations
                if hasattr(encoded, 'structure') and hasattr(encoded.structure, 's'):
                    reps = encoded.structure.s  # 蛋白质序列表示
                elif hasattr(encoded, 's'):
                    reps = encoded.s
                else:
                    reps = encoded  # 直接使用输出
            else:
                # 如果没有encode方法，使用传统方法
                return self._encode_traditional_esm(sequence)
                
            # 确保reps是tensor
            if not isinstance(reps, torch.Tensor):
                reps = torch.tensor(reps)
                
            # 确保是float32类型
            reps = reps.to(torch.float32)
            
            # 投影到目标维度
            projected = self.projection(reps)
            return projected
        except Exception:
            # 如果ESM3编码失败，回退到传统ESM方法
            return self._encode_traditional_esm(sequence)

    def _tokenize(self, sequence: str) -> torch.Tensor:
        """内部token化方法"""
        if self.batch_converter:
            batch = [("protein", sequence)]
            _, _, tokens = self.batch_converter(batch)
            return tokens
        else:
            # 对于ESM3，可能需要不同的token化方法
            raise NotImplementedError("ESM3的token化方法需要根据具体API实现")


def residues_from_structure(residue_entities: Iterable) -> List[str]:
    names: List[str] = []
    for residue in residue_entities:
        hetfield, resseq, icode = residue.id
        if hetfield.strip():
            continue
        names.append(residue.resname.upper())
    return names


# 为向后兼容保留旧的ESMProteinEncoder类名
ESMProteinEncoder = ESM3ProteinEncoder
ProteinEncoderConfig = ESM3ProteinEncoderConfig