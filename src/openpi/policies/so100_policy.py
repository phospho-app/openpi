import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_so100_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(12),
        "observation/images.main": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images.secondary_0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images.secondary_1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class S0100Inputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI05

    image_keys: list[str] = dataclasses.field(default_factory=lambda: ["observation.images.main"])

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(np.asarray(data["observation/state"]), self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        images = {}
        image_names = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        for image in self.image_keys[:3]:  # Only take up to 3 images
            format_key = image.replace("observation.", "observation/")
            if format_key not in data:
                raise ValueError(
                    f"Expected image key {format_key} in data but not found. Available keys: {list(data.keys())}"
                )
            images[image_names.pop(0)] = _parse_image(data[format_key])
        if len(images) < 3:
            for _ in range(3 - len(images)):
                images[image_names.pop(0)] = np.zeros_like(images["base_0_rgb"])

        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class S0100Outputs(transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data: dict) -> dict:
        # Make sure to only return the appropriate number of actions here
        # 6 for 1 robot, 12 for 2
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
