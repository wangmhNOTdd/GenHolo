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
class ProteinEncoderConfig:
    model_name: str
    device: str
    fp16: bool
    chunk_size: int
    proj_in: int
    proj_out: int
    proj_checkpoint: str | None = None


class ESMProteinEncoder:
    def __init__(self, cfg: ProteinEncoderConfig) -> None:
        self.cfg = cfg
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(cfg.model_name)
        self.model.eval()
        self.model = self.model.to(cfg.device)
        if cfg.fp16:
            self.model = self.model.half()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.cls_token = self.alphabet.cls_idx
        self.eos_token = self.alphabet.eos_idx
        self.padding_idx = self.alphabet.padding_idx
        self.projection = torch.nn.Linear(cfg.proj_in, cfg.proj_out)
        if cfg.proj_checkpoint:
            state = torch.load(cfg.proj_checkpoint, map_location="cpu")
            self.projection.load_state_dict(state)
        self.projection.eval()

    def encode(self, residues: Sequence[str]) -> torch.Tensor:
        sequence = "".join(AA_THREE_TO_ONE.get(res, "X") for res in residues)
        tokens = self._tokenize(sequence)
        with torch.no_grad():
            model_out = self.model(tokens.to(self.cfg.device), repr_layers=[self.model.num_layers])
            reps = model_out["representations"][self.model.num_layers]
            reps = reps[:, 1 : sequence.__len__() + 1]
            reps = reps.to(torch.float32)
            projected = self.projection(reps)
        return projected.squeeze(0)

    def _tokenize(self, sequence: str) -> torch.Tensor:
        batch = [("protein", sequence)]
        _, _, tokens = self.batch_converter(batch)
        return tokens


def residues_from_structure(residue_entities: Iterable) -> List[str]:
    names: List[str] = []
    for residue in residue_entities:
        hetfield, resseq, icode = residue.id
        if hetfield.strip():
            continue
        names.append(residue.resname.upper())
    return names
