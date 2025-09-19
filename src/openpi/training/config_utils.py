"""Shared utilities for creating dynamic training configurations."""

import dataclasses
from typing import Any

import openpi.training.config as _config


def apply_override(config: _config.TrainConfig, key: str, value: Any):
    """Recreate dataclass configs with updated values using dot notation."""
    parts = key.split(".")
    if len(parts) == 1:
        # Replace the whole object
        if dataclasses.is_dataclass(config):
            return dataclasses.replace(config, **{parts[0]: value})
        setattr(config, parts[0], value)
        return config

    # Recurse into nested configs
    attr = getattr(config, parts[0])
    updated = apply_override(attr, ".".join(parts[1:]), value)

    if dataclasses.is_dataclass(config):
        return dataclasses.replace(config, **{parts[0]: updated})
    setattr(config, parts[0], updated)
    return config
