import numpy as np

from openpi.models import model as _model
from openpi.policies import deepcybo_lite_policy


def test_deepcybo_lite_inputs_pi05():
    transform = deepcybo_lite_policy.DeepCyboLiteInputs(model_type=_model.ModelType.PI05)
    data = deepcybo_lite_policy.make_deepcybo_lite_example()
    data["actions"] = np.zeros((4, deepcybo_lite_policy.ACTION_DIM), dtype=np.float32)

    output = transform(data)

    assert output["state"].shape == (deepcybo_lite_policy.ACTION_DIM,)
    assert output["actions"].shape == (4, deepcybo_lite_policy.ACTION_DIM)
    assert set(output["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert all(output["image_mask"].values())


def test_deepcybo_lite_inputs_missing_wrist_image_masks_padding():
    transform = deepcybo_lite_policy.DeepCyboLiteInputs(model_type=_model.ModelType.PI05)
    data = deepcybo_lite_policy.make_deepcybo_lite_example()
    del data["images"]["image_wrist_right"]

    output = transform(data)

    assert output["image_mask"]["right_wrist_0_rgb"] == np.False_
    assert np.all(output["image"]["right_wrist_0_rgb"] == 0)


def test_deepcybo_lite_outputs():
    transform = deepcybo_lite_policy.DeepCyboLiteOutputs()
    data = {"actions": np.zeros((8, 32), dtype=np.float32)}

    output = transform(data)

    assert output["actions"].shape == (8, deepcybo_lite_policy.ACTION_DIM)
