import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

JOINT_NAMES: tuple[str, ...] = (
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow_pitch",
    "left_wrist_yaw",
    "left_wrist_roll",
    "left_wrist_pitch",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow_pitch",
    "right_wrist_yaw",
    "right_wrist_roll",
    "right_wrist_pitch",
    "left_gripper",
    "right_gripper",
)

ACTION_DIM = len(JOINT_NAMES)


def make_deepcybo_lite_example() -> dict:
    """Creates a random input example for the DeepCybo Lite policy."""
    return {
        "state": np.random.rand(ACTION_DIM).astype(np.float32),
        "images": {
            "image_head": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "image_wrist_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "image_wrist_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _ensure_action_dim(name: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected {name} last dimension to be {ACTION_DIM}, got shape {value.shape}")
    return value


@dataclasses.dataclass(frozen=True)
class DeepCyboLiteInputs(transforms.DataTransformFn):
    """Inputs for the DeepCybo Lite policy.

    Expected inputs after training repack or from the robot runtime:
    - images: dict with image_head, image_wrist_left, image_wrist_right.
    - state: [16] in JOINT_NAMES order.
    - actions: [action_horizon, 16] in JOINT_NAMES order, training only.
    - prompt: optional language instruction.
    """

    model_type: _model.ModelType

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("image_head", "image_wrist_left", "image_wrist_right")

    def __call__(self, data: dict) -> dict:
        state = _ensure_action_dim("state", data["state"])
        in_images = data["images"]
        unknown_cameras = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unknown_cameras:
            raise ValueError(f"Unexpected DeepCybo Lite cameras: {tuple(sorted(unknown_cameras))}")
        if "image_head" not in in_images:
            raise ValueError("DeepCybo Lite input requires image_head")

        head_image = _parse_image(in_images["image_head"])
        left_wrist_image = _parse_image(in_images["image_wrist_left"]) if "image_wrist_left" in in_images else None
        right_wrist_image = _parse_image(in_images["image_wrist_right"]) if "image_wrist_right" in in_images else None

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                images = {
                    "base_0_rgb": head_image,
                    "left_wrist_0_rgb": left_wrist_image if left_wrist_image is not None else np.zeros_like(head_image),
                    "right_wrist_0_rgb": right_wrist_image
                    if right_wrist_image is not None
                    else np.zeros_like(head_image),
                }
                image_masks = {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.bool_(left_wrist_image is not None),
                    "right_wrist_0_rgb": np.bool_(right_wrist_image is not None),
                }
            case _model.ModelType.PI0_FAST:
                images = {
                    "base_0_rgb": head_image,
                    "base_1_rgb": left_wrist_image if left_wrist_image is not None else np.zeros_like(head_image),
                    "wrist_0_rgb": right_wrist_image if right_wrist_image is not None else np.zeros_like(head_image),
                }
                # FAST models do not mask out padding image slots in the existing openpi examples.
                image_masks = {
                    "base_0_rgb": np.True_,
                    "base_1_rgb": np.True_,
                    "wrist_0_rgb": np.True_,
                }
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        if "actions" in data:
            inputs["actions"] = _ensure_action_dim("actions", data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class DeepCyboLiteOutputs(transforms.DataTransformFn):
    """Outputs for the DeepCybo Lite policy."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}
