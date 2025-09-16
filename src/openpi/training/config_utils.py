"""Shared utilities for creating dynamic training configurations."""

import dataclasses
import logging
from pathlib import Path

import datasets
import numpy as np

import openpi.training.config as _config
import openpi.transforms as transforms


@dataclasses.dataclass(frozen=True)
class CustomDataConfig(_config.DataConfigFactory):
    """Flexible data config for custom robot setups."""

    # Camera configuration
    image_keys: list[str] = dataclasses.field(default_factory=list)  # Dataset image keys

    # Action configuration
    action_key: str = "action"  # Dataset action key
    state_key: str = "observation/state"  # Dataset state key

    def create(self, assets_dirs: Path, model_config: _config._model.BaseModelConfig) -> _config.DataConfig:
        # Create image mappings - map dataset keys to standard observation format
        image_mapping = {}
        for i, dataset_key in enumerate(self.image_keys):
            obs_key = "main" if i == 0 else f"secondary_{i - 1}"
            image_mapping[dataset_key] = obs_key

        # Create repack mapping to standardize dataset format
        repack_mapping = {
            **image_mapping,
            self.state_key: "state",
            self.action_key: "actions",
            "prompt": "prompt",
        }

        # Create repack transform using the transforms module from config
        repack_transforms = transforms.Group(inputs=[transforms.RepackTransform(repack_mapping)])

        # Use basic identity transforms for data and model processing
        # User can extend this by creating custom transform classes if needed
        data_transforms = transforms.Group(inputs=[], outputs=[])
        model_transforms = _config.ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=(self.action_key,),
        )


def inspect_dataset_keys(
    repo_id: str, action_key: str, state_key: str, image_keys: list[str]
) -> tuple[str, str, list[str]]:
    """Inspect dataset to automatically detect keys if not provided."""
    # Download a sample of the dataset to inspect keys
    dataset = datasets.load_dataset(repo_id, split="train", streaming=True, data_files="meta/info.json")
    sample = next(iter(dataset))
    features = sample.get("features", None)
    if not features:
        raise ValueError("Dataset meta/info.json does not contain 'features' key.")

    logging.debug(f"Found the following dataset keys: {list(features)}")

    # Auto-detect image keys if not provided
    detected_image_keys = image_keys if image_keys else [k for k in features if "image" in k]
    if not detected_image_keys:
        raise ValueError("No image keys found in the dataset. Please specify the image keys flag.")

    # Auto-detect action key if not found
    detected_action_key = (
        action_key if action_key in features else next((k for k in features if "action" in k), action_key)
    )
    if detected_action_key not in features:
        raise ValueError(
            f"Action key '{detected_action_key}' not found in the dataset. Please specify the action key flag."
        )

    # Auto-detect state key if not found
    detected_state_key = state_key if state_key in features else next((k for k in features if "state" in k), state_key)
    if detected_state_key not in features:
        raise ValueError(
            f"State key '{detected_state_key}' not found in the dataset. Please specify the state key flag."
        )

    return detected_action_key, detected_state_key, detected_image_keys


def create_dynamic_config(
    repo_id: str,
    exp_name: str,
    image_keys: list[str],
    action_key: str = "action",
    state_key: str = "observation/state",
    action_dim: int = 6,
    action_horizon: int = 10,
    batch_size: int = 64,
    num_train_steps: int = 30000,
    checkpoint_base_dir: str = "./checkpoints",
    wandb_enabled: bool = True,
) -> _config.TrainConfig:
    """Create a dynamic training config for Pi0.5 LoRA fine-tuning."""

    # Use Pi0.5 with LoRA configuration
    model_config = _config.pi0_config.Pi0Config(
        pi05=True,
        action_dim=action_dim,
        action_horizon=action_horizon,
        paligemma_variant="gemma_2b_lora",  # Use LoRA variant
        action_expert_variant="gemma_300m_lora",  # Use LoRA variant
    )

    # Pi0.5 base model checkpoint
    weight_loader_path = "gs://openpi-assets/checkpoints/pi05_base/params"

    # Create custom data config
    data_factory = CustomDataConfig(
        repo_id=repo_id,
        image_keys=image_keys,
        action_key=action_key,
        state_key=state_key,
    )

    return _config.TrainConfig(
        name="custom_pi05_lora",
        exp_name=exp_name,
        model=model_config,
        data=data_factory,
        weight_loader=_config.weight_loaders.CheckpointWeightLoader(weight_loader_path),
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        checkpoint_base_dir=checkpoint_base_dir,
        freeze_filter=model_config.get_freeze_filter(),
        ema_decay=None,
        wandb_enabled=wandb_enabled,
    )


def prepare_custom_config_from_args(
    repo_id: str,
    action_dim: int,
    action_horizon: int,
    batch_size: int,
    num_train_steps: int,
    image_keys: str,
    action_key: str,
    state_key: str,
    wandb_enabled: bool,
    checkpoint_base_dir: str | None = None,
) -> _config.TrainConfig:
    """Prepare a custom config from command line arguments with dataset inspection."""

    image_keys_list = [key.strip() for key in image_keys.split(",") if key.strip()]

    # Inspect dataset and auto-detect keys
    detected_action_key, detected_state_key, detected_image_keys = inspect_dataset_keys(
        repo_id, action_key, state_key, image_keys_list
    )

    # Generate experiment name with random suffix
    random_suffix = f"_{np.random.randint(0, 10000):04d}"
    exp_name = repo_id.replace("/", "_") + random_suffix

    # Use provided checkpoint base dir or default based on repo
    final_checkpoint_base_dir = checkpoint_base_dir or f"./checkpoints/{repo_id}"

    return create_dynamic_config(
        repo_id=repo_id,
        exp_name=exp_name,
        image_keys=detected_image_keys,
        action_key=detected_action_key,
        state_key=detected_state_key,
        action_dim=action_dim,
        action_horizon=action_horizon,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        checkpoint_base_dir=final_checkpoint_base_dir,
        wandb_enabled=wandb_enabled,
    )
