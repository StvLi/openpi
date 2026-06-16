from openpi.policies import deepcybo_lite_policy
from openpi.training import config as _config
import openpi.transforms as _transforms


def test_pi05_deepcybo_lite_config_registered():
    config = _config.get_config("pi05_deepcybo_lite")
    data_config = config.data.create(config.assets_dirs, config.model)

    assert data_config.repo_id == "local/deepcybo_lite_bilateral"
    assert data_config.action_sequence_keys == ("action",)
    assert data_config.prompt_from_task
    assert any(isinstance(transform, _transforms.RepackTransform) for transform in data_config.repack_transforms.inputs)
    assert any(
        isinstance(transform, deepcybo_lite_policy.DeepCyboLiteInputs)
        for transform in data_config.data_transforms.inputs
    )


def test_pi05_deepcybo_lite_low_mem_config_registered():
    config = _config.get_config("pi05_deepcybo_lite_low_mem_finetune")

    assert config.ema_decay is None
    assert config.model.action_dim == 32
    assert config.model.action_horizon == 16
