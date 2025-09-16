"""Shared utilities for creating dynamic training configurations."""

import ast
import dataclasses

import openpi.training.config as _config


def _auto_cast(value: str):
    """Try to interpret strings as bool, int, float, list, or leave as str."""
    # Handle boolean values
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    
    # Handle lists - support both Python list syntax and comma-separated values
    if value.startswith('[') and value.endswith(']'):
        try:
            # Try to parse as Python literal (handles nested structures)
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Fall back to simple comma-separated parsing
            inner = value[1:-1].strip()
            if not inner:  # Empty list
                return []
            items = [item.strip().strip('"\'') for item in inner.split(',')]
            return [_auto_cast(item) for item in items]
    
    # Handle comma-separated values (interpret as list if multiple values)
    if ',' in value:
        items = [item.strip().strip('"\'') for item in value.split(',')]
        if len(items) > 1:
            return [_auto_cast(item) for item in items]
    
    # Handle numeric values
    try:
        # Try integer first
        if '.' not in value and 'e' not in value.lower():
            return int(value)
    except ValueError:
        pass
    
    try:
        # Try float
        return float(value)
    except ValueError:
        pass
    
    # Handle quoted strings
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    
    # Return as string
    return value


def apply_override(config: _config.TrainConfig, key: str, value: str):
    """Recreate dataclass configs with updated values using dot notation."""
    parts = key.split(".")
    if len(parts) == 1:
        # Replace the whole object
        if dataclasses.is_dataclass(config):
            return dataclasses.replace(config, **{parts[0]: _auto_cast(value)})
        else:
            setattr(config, parts[0], _auto_cast(value))
            return config

    # Recurse into nested configs
    attr = getattr(config, parts[0])
    updated = apply_override(attr, ".".join(parts[1:]), value)

    if dataclasses.is_dataclass(config):
        return dataclasses.replace(config, **{parts[0]: updated})
    else:
        setattr(config, parts[0], updated)
        return config