"""
测试 EmbodimentConfig 和 Registry 系统

运行方式:
    python -m pytest tests/test_embodiment_config.py -v
    或
    python tests/test_embodiment_config.py  # 直接运行
"""

import tempfile
from pathlib import Path

from openpi.shared.embodiment_config import (
    EmbodimentConfig,
    EmbodimentKey,
    EmbodimentRegistry,
    get_default_config,
    get_training_config,
)


def test_embodiment_key_creation():
    """测试 EmbodimentKey 的创建和属性访问"""
    key = EmbodimentKey(
        robot_type="franka",
        dof=7,
        action_space="cartesian",
        state_space="cartesian",
        coordinate_frame="base",
        image_crop=False,
        image_rotation=False,
        image_flip=False,
        camera_viewpoint_id="default",
    )

    # 测试字段访问
    assert key.robot_type == "franka"
    assert key.dof == 7
    assert key.action_space == "cartesian"

    # 测试打印（应该显示字段名）
    key_str = str(key)
    assert "robot_type='franka'" in key_str
    assert "dof=7" in key_str

    print(f"✓ EmbodimentKey creation: {key}")


def test_embodiment_key_equality():
    """测试 EmbodimentKey 的相等性比较"""
    key1 = EmbodimentKey(
        robot_type="franka",
        dof=7,
        action_space="cartesian",
        state_space="cartesian",
        coordinate_frame="base",
        image_crop=False,
        image_rotation=False,
        image_flip=False,
        camera_viewpoint_id="default",
    )

    key2 = EmbodimentKey(
        robot_type="franka",
        dof=7,
        action_space="cartesian",
        state_space="cartesian",
        coordinate_frame="base",
        image_crop=False,
        image_rotation=False,
        image_flip=False,
        camera_viewpoint_id="default",
    )

    key3 = EmbodimentKey(
        robot_type="franka",
        dof=7,
        action_space="cartesian",
        state_space="cartesian",
        coordinate_frame="base",
        image_crop=True,  # 不同!
        image_rotation=False,
        image_flip=False,
        camera_viewpoint_id="default",
    )

    assert key1 == key2, "相同配置的 key 应该相等"
    assert key1 != key3, "不同配置的 key 应该不相等"
    assert hash(key1) == hash(key2), "相同配置的 hash 应该相同"

    print("✓ EmbodimentKey equality and hashing")


def test_embodiment_config_to_key():
    """测试 EmbodimentConfig 转换为 EmbodimentKey"""
    config = EmbodimentConfig(
        robot_type="franka",
        image_crop=True,
        image_rotation=False,
        _color_jitter=True,  # 不应该影响 key
    )

    key = config.to_key()

    assert key.robot_type == "franka"
    assert key.image_crop is True
    assert key.image_rotation is False

    # 验证外观增强不影响 key
    config_with_color = EmbodimentConfig(
        robot_type="franka",
        image_crop=True,
        image_rotation=False,
        _color_jitter=False,  # 不同的外观增强
    )

    key_with_color = config_with_color.to_key()
    assert key == key_with_color, "外观增强不应该影响 EmbodimentKey"

    print("✓ EmbodimentConfig.to_key() - 外观增强正确地被忽略")


def test_embodiment_config_readable_string():
    """测试可读字符串生成"""
    config1 = EmbodimentConfig(
        robot_type="franka",
        image_crop=False,
        image_rotation=False,
    )
    assert config1.to_readable_string() == "franka_7dof_cartesian_base_noaug_camdefault"

    config2 = EmbodimentConfig(
        robot_type="franka",
        image_crop=True,
        image_rotation=True,
    )
    assert "crop" in config2.to_readable_string()
    assert "rot" in config2.to_readable_string()

    print(f"✓ Readable strings: {config1.to_readable_string()}, {config2.to_readable_string()}")


def test_registry_auto_mode():
    """测试 Registry 的自动注册模式"""
    registry = EmbodimentRegistry(mode="auto")

    # 注册第一个配置
    config1 = EmbodimentConfig(robot_type="franka", image_crop=False)
    idx1 = registry.get_or_register(config1)
    assert idx1 == 0, "第一个配置应该得到 index 0"

    # 注册不同的配置
    config2 = EmbodimentConfig(robot_type="franka", image_crop=True)
    idx2 = registry.get_or_register(config2)
    assert idx2 == 1, "新配置应该得到 index 1"

    # 重复注册相同配置（应该返回相同 index）
    config3 = EmbodimentConfig(robot_type="franka", image_crop=False)
    idx3 = registry.get_or_register(config3)
    assert idx3 == 0, "相同配置应该复用 index 0"

    # 外观增强不影响 index
    config4 = EmbodimentConfig(
        robot_type="franka", image_crop=False, _color_jitter=True  # 外观增强
    )
    idx4 = registry.get_or_register(config4)
    assert idx4 == 0, "外观增强不应该影响 index"

    assert len(registry) == 2, "应该有 2 个不同的 embodiment"

    print(f"✓ Registry auto mode: {len(registry)} embodiments registered")
    print(registry.summary())


def test_registry_manual_mode():
    """测试 Registry 的手动模式"""
    registry = EmbodimentRegistry(mode="manual")

    # 预先注册一个配置
    config1 = EmbodimentConfig(robot_type="franka", image_crop=False)
    registry.mode = "auto"  # 临时切换到 auto 注册
    idx1 = registry.get_or_register(config1)
    registry.mode = "manual"  # 切回 manual

    # 尝试获取已注册的配置（应该成功）
    idx_again = registry.get_or_register(config1)
    assert idx_again == idx1

    # 尝试注册新配置（应该失败）
    config2 = EmbodimentConfig(robot_type="ur5")
    try:
        registry.get_or_register(config2)
        assert False, "Manual 模式下应该拒绝新配置"
    except ValueError as e:
        assert "manual mode" in str(e).lower()
        print(f"✓ Registry manual mode correctly rejects unknown config: {e}")


def test_registry_save_load():
    """测试 Registry 的保存和加载"""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "registry.json"

        # 创建 registry 并注册一些配置
        registry1 = EmbodimentRegistry(mode="auto")
        config1 = EmbodimentConfig(robot_type="franka", image_crop=False)
        config2 = EmbodimentConfig(robot_type="franka", image_crop=True)
        config3 = EmbodimentConfig(robot_type="ur5", dof=6)

        idx1 = registry1.get_or_register(config1)
        idx2 = registry1.get_or_register(config2)
        idx3 = registry1.get_or_register(config3)

        # 保存
        registry1.save(registry_path)
        assert registry_path.exists(), "Registry 文件应该被创建"

        # 加载到新的 registry
        registry2 = EmbodimentRegistry(mode="auto")
        registry2.load(registry_path)

        # 验证加载的内容
        assert len(registry2) == 3, "应该加载 3 个 embodiment"

        # 验证 index 一致
        assert registry2.get_or_register(config1) == idx1
        assert registry2.get_or_register(config2) == idx2
        assert registry2.get_or_register(config3) == idx3

        print(f"✓ Registry save/load: {registry_path}")


def test_registry_merge():
    """测试 Registry 的合并"""
    registry1 = EmbodimentRegistry(mode="auto")
    registry1.get_or_register(EmbodimentConfig(robot_type="franka"))

    registry2 = EmbodimentRegistry(mode="auto")
    registry2.get_or_register(EmbodimentConfig(robot_type="ur5", dof=6))
    registry2.get_or_register(EmbodimentConfig(robot_type="aloha", dof=14))

    # 合并
    registry1.merge(registry2)

    assert len(registry1) == 3, "合并后应该有 3 个 embodiment"

    print(f"✓ Registry merge: {len(registry1)} embodiments")


def test_factory_functions():
    """测试工厂函数"""
    # 默认配置
    config_default = get_default_config("franka")
    assert config_default.robot_type == "franka"
    assert config_default.image_crop is False
    assert config_default.image_rotation is False

    # 训练配置
    config_train = get_training_config("franka", enable_geometric_aug=True)
    assert config_train.robot_type == "franka"
    assert config_train.image_crop is True
    assert config_train.image_rotation is True
    assert config_train._color_jitter is True

    # 训练配置（不启用几何增强）
    config_train_no_geo = get_training_config("franka", enable_geometric_aug=False)
    assert config_train_no_geo.image_crop is False
    assert config_train_no_geo.image_rotation is False
    assert config_train_no_geo._color_jitter is True  # 仍然有外观增强

    print("✓ Factory functions")


def test_real_world_scenario():
    """测试真实场景：同一数据集的不同增强策略"""
    registry = EmbodimentRegistry(mode="auto")

    # 场景 1: 推理时不增强
    config_inference = EmbodimentConfig(
        robot_type="franka", image_crop=False, image_rotation=False
    )
    w_idx_inference = registry.get_or_register(config_inference)

    # 场景 2: 训练时几何增强
    config_train_geo = EmbodimentConfig(
        robot_type="franka", image_crop=True, image_rotation=True
    )
    w_idx_train_geo = registry.get_or_register(config_train_geo)

    # 场景 3: 训练时只有外观增强（与推理共享 W）
    config_train_appearance = EmbodimentConfig(
        robot_type="franka",
        image_crop=False,
        image_rotation=False,
        _color_jitter=True,  # 外观增强
        _brightness_aug=True,
    )
    w_idx_train_appearance = registry.get_or_register(config_train_appearance)

    # 验证
    assert w_idx_inference != w_idx_train_geo, "几何增强应该使用不同的 W"
    assert w_idx_inference == w_idx_train_appearance, "外观增强应该共享 W"

    print("✓ Real-world scenario:")
    print(f"  Inference W index: {w_idx_inference}")
    print(f"  Train (geo aug) W index: {w_idx_train_geo}")
    print(f"  Train (appearance aug) W index: {w_idx_train_appearance}")
    print(registry.summary())


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Testing EmbodimentConfig System")
    print("=" * 60)

    tests = [
        test_embodiment_key_creation,
        test_embodiment_key_equality,
        test_embodiment_config_to_key,
        test_embodiment_config_readable_string,
        test_registry_auto_mode,
        test_registry_manual_mode,
        test_registry_save_load,
        test_registry_merge,
        test_factory_functions,
        test_real_world_scenario,
    ]

    for test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_func.__name__}")
        print(f"{'='*60}")
        try:
            test_func()
            print(f"✅ {test_func.__name__} PASSED")
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
