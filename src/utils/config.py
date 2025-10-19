import os
from copy import deepcopy
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover - 导入错误时直接提示依赖
    raise ImportError("请先安装 PyYAML，命令: pip install pyyaml") from exc


def _resolve_item(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, str) and "${" in value:
        resolved = value
        for _ in range(5):  # 最多展开 5 层嵌套
            start = resolved.find("${")
            if start == -1:
                break
            end = resolved.find("}", start)
            if end == -1:
                break
            key = resolved[start + 2:end]
            replacement = _lookup_key(context, key)
            if replacement is None:
                raise KeyError(f"未能在配置中解析键 '{key}'")
            resolved = resolved[:start] + str(replacement) + resolved[end + 1:]
        return os.path.expanduser(resolved)
    if isinstance(value, dict):
        return {k: _resolve_item(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_item(v, context) for v in value]
    return value


def _lookup_key(context: Dict[str, Any], dotted_key: str) -> Any:
    parts = dotted_key.split(".")
    current = context
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    context = deepcopy(raw)
    # 先展开环境变量
    for key, value in raw.items():
        context[key] = _resolve_item(value, raw)
    return context
