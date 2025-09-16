"""Compute normalization statistics for a dataset.

USAGE:
uv run scripts/compute_norm_stats.py compute-norm-custom \
    --repo-id "your-username/your-dataset" \
    --action-dim 12 \
    --action-horizon 10 \
    --batch-size 64

Or with predefined config:
uv run scripts/compute_norm_stats.py compute-norm-with-config pi0_libero
"""

import numpy as np
import tqdm
import typer

import openpi.models.model as _model
import openpi.shared.normalize as normalize
from openpi.training import config_utils
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
):
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
):
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def compute_with_config(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    repo_id_path = data_config.repo_id or "unknown"
    output_path = config.assets_dirs / repo_id_path
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


app = typer.Typer()


@app.command()
def compute_norm_with_config(config_name: str, max_frames: int | None = None):
    """Compute normalization stats with a predefined configuration."""
    compute_with_config(config_name, max_frames)


@app.command()
def compute_norm_custom(
    repo_id: str = typer.Option(..., help="HuggingFace dataset repository ID"),
    action_dim: int = typer.Option(7, help="Action dimension"),
    action_horizon: int = typer.Option(10, help="Action horizon"),
    batch_size: int = typer.Option(64, help="Batch size"),
    image_keys: str = typer.Option("", help="Comma-separated list of image keys in the dataset"),
    action_key: str = typer.Option("action", help="Action key in the dataset"),
    state_key: str = typer.Option("observation/state", help="State key in the dataset"),
    max_frames: int | None = typer.Option(None, help="Maximum number of frames to process"),
):
    """Compute normalization stats for Pi0.5 with custom arguments."""

    config = config_utils.prepare_custom_config_from_args(
        repo_id=repo_id,
        action_dim=action_dim,
        action_horizon=action_horizon,
        batch_size=batch_size,
        num_train_steps=0,  # Not used for norm stats
        image_keys=image_keys,
        action_key=action_key,
        state_key=state_key,
    )

    compute_with_config_object(config, max_frames)


def compute_with_config_object(config: _config.TrainConfig, max_frames: int | None = None):
    """Helper function to compute normalization stats with a config object."""
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    repo_id_path = data_config.repo_id or "unknown"
    output_path = config.assets_dirs / repo_id_path
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    app()
