"""YAML config loading with dotted-key overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply `a.b.c=value` overrides; values are parsed as YAML scalars."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.path=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        node = cfg
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = load_yaml(path)
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = deep_merge(load_config(Path(path).parent / parent), cfg)
    return apply_overrides(cfg, overrides or [])


def resolve(path: str | Path, base: Path = REPO_ROOT) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
