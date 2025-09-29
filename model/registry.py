"""Lightweight module registry for decoder backbones."""
from __future__ import annotations

from typing import Callable, Dict, Type

import torch.nn as nn


_DECODER_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_decoder(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Decorator to register a decoder class under ``name``.

    Args:
        name: Unique string identifier for the decoder.
    """

    def _register(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _DECODER_REGISTRY and _DECODER_REGISTRY[name] is not cls:
            raise ValueError(f"Decoder '{name}' already registered")
        _DECODER_REGISTRY[name] = cls
        return cls

    return _register


def get_decoder_cls(name: str) -> Type[nn.Module]:
    try:
        return _DECODER_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_DECODER_REGISTRY)) or "<none>"
        raise KeyError(f"Decoder '{name}' not found. Available: {available}") from exc


def build_decoder(name: str, **kwargs) -> nn.Module:
    decoder_cls = get_decoder_cls(name)
    return decoder_cls(**kwargs)


def list_decoders() -> Dict[str, Type[nn.Module]]:
    return dict(_DECODER_REGISTRY)
