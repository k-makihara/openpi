import dataclasses

import numpy as np

from openpi.policies import umi_original_policy
from openpi.training import config as training_config


def _pose(x: float, y: float, z: float) -> np.ndarray:
    return np.asarray([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def test_gripper_fill_unwrap_and_bound_alignment():
    values = np.asarray([0.0, 0.0, 6.0, 0.0, 0.4, 0.0], dtype=np.float32)

    filled = umi_original_policy._forward_fill_zeros(values)  # noqa: SLF001

    np.testing.assert_allclose(
        filled,
        np.asarray([6.0 - (0.4 + 2.0 * np.pi - 6.28)] * 4 + [6.28, 6.28]),
        rtol=1e-6,
        atol=1e-6,
    )

    for value, expected in ((2.9, 6.28), (3.0, 6.28), (3.1, 5.41)):
        np.testing.assert_allclose(
            umi_original_policy._forward_fill_zeros(np.asarray([value])),  # noqa: SLF001
            np.asarray([expected]),
            rtol=1e-6,
            atol=1e-6,
        )

    np.testing.assert_allclose(
        umi_original_policy._forward_fill_zeros(np.zeros(3, dtype=np.float32)),  # noqa: SLF001
        np.full(3, 6.28, dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        umi_original_policy._forward_fill_zeros(np.asarray([5.0, 5.8])),  # noqa: SLF001
        np.asarray([5.41, 6.21]),
        rtol=1e-6,
        atol=1e-6,
    )


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
    np.testing.assert_allclose(result["state"][[9, 19, 29, 39]], np.full(4, 6.08), atol=1e-6)
    np.testing.assert_allclose(result["actions"][:, [9, 19]], [[6.18, 6.18], [6.28, 6.28]], atol=1e-6)
    np.testing.assert_allclose(result["state"][-18:-15], np.asarray([-1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(result["state"][-15:-9], np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(result["state"][-9:-6], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(result["state"][-6:], np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32))

    preprocessed = dataclasses.replace(transform, gripper_preprocessed=True)
    preprocessed_result = preprocessed(data)
    np.testing.assert_allclose(preprocessed_result["state"][[9, 19, 29, 39]], [0.5, 0.5, 0.2, 0.2])
    np.testing.assert_allclose(preprocessed_result["actions"][:, [9, 19]], [[0.6, 0.3], [0.7, 0.4]])


def test_dual_arm_relative_state_config_registered():
    config = training_config.get_config("pi05_umi_original_dual_arm_relative_state_h16_bs32_30k")
    data_config = config.data.create(config.assets_dirs, config.model)

    assert config.model.action_dim == 32
    assert isinstance(config.data, training_config.LeRobotUmiOriginalDualArmRelativeStateDataConfig)
    assert isinstance(
        data_config.data_transforms.inputs[0],
        umi_original_policy.UmiOriginalDualArmRelativeStateInputs,
    )
