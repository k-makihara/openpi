import numpy as np

from openpi.policies import umi_original_policy
from openpi.training import config as training_config


def _pose(x: float, y: float, z: float) -> np.ndarray:
    return np.asarray([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def test_dual_arm_relative_state_inputs():
    transform = umi_original_policy.UmiOriginalDualArmRelativeStateInputs(
        history_steps=2,
        action_horizon_steps=2,
        action_pose_target="relative",
    )

    data = {
        "left_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "right_image": np.ones((8, 8, 3), dtype=np.uint8),
        "left_pose_seq": np.asarray(
            [_pose(0.0, 0.0, 0.0), _pose(0.0, 0.0, 0.0), _pose(0.1, 0.0, 0.0), _pose(0.2, 0.0, 0.0)],
            dtype=np.float32,
        ),
        "right_pose_seq": np.asarray(
            [_pose(1.0, 0.0, 0.0), _pose(1.0, 0.0, 0.0), _pose(1.1, 0.0, 0.0), _pose(1.3, 0.0, 0.0)],
            dtype=np.float32,
        ),
        "left_gripper_seq": np.asarray([0.5, 0.5, 0.6, 0.7], dtype=np.float32),
        "right_gripper_seq": np.asarray([0.2, 0.2, 0.3, 0.4], dtype=np.float32),
        "prompt": "handover object",
    }

    result = transform(data)

    assert result["state"].shape == (58,)
    assert result["actions"].shape == (2, 20)
    np.testing.assert_array_equal(result["image"]["base_0_rgb"], np.zeros((8, 8, 3), dtype=np.uint8))
    np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], data["left_image"])
    np.testing.assert_array_equal(result["image"]["right_wrist_0_rgb"], data["right_image"])
    assert not result["image_mask"]["base_0_rgb"]
    assert result["image_mask"]["left_wrist_0_rgb"]
    assert result["image_mask"]["right_wrist_0_rgb"]
    np.testing.assert_allclose(result["state"][-18:-15], np.asarray([-1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(result["state"][-15:-9], np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(result["state"][-9:-6], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(result["state"][-6:], np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32))


def test_dual_arm_relative_state_config_registered():
    config = training_config.get_config("pi05_umi_original_dual_arm_relative_state_h16_bs32_30k")
    data_config = config.data.create(config.assets_dirs, config.model)

    assert config.model.action_dim == 32
    assert isinstance(config.data, training_config.LeRobotUmiOriginalDualArmRelativeStateDataConfig)
    assert isinstance(
        data_config.data_transforms.inputs[0],
        umi_original_policy.UmiOriginalDualArmRelativeStateInputs,
    )
