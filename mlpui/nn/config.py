"""
mlpui/nn/config.py

配置文件解析
============

支持 YAML/JSON 配置文件
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import json


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    从文件加载配置

    支持 YAML 和 JSON 格式

    Args:
        path: 配置文件路径

    Returns:
        配置字典
    """
    path = Path(path)

    if path.suffix in ['.yaml', '.yml']:
        try:
            import yaml
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")

    elif path.suffix == '.json':
        with open(path, 'r') as f:
            return json.load(f)

    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def save_config(config: Dict[str, Any], path: Union[str, Path]):
    """
    保存配置到文件

    Args:
        config: 配置字典
        path: 保存路径
    """
    path = Path(path)

    if path.suffix in ['.yaml', '.yml']:
        try:
            import yaml
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        except ImportError:
            raise ImportError("PyYAML not installed.")

    elif path.suffix == '.json':
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置的有效性

    Args:
        config: 配置字典

    Returns:
        是否有效

    Raises:
        ValueError: 配置无效时
    """
    from mlpui.inputs import INPUT_REGISTRY
    from mlpui.models import MODEL_REGISTRY
    from mlpui.outputs import OUTPUT_REGISTRY

    # 检查必需字段
    if 'model' not in config:
        raise ValueError("Config must contain 'model' section")

    # 检查 input 类型
    input_config = config.get('input', {})
    input_type = input_config.get('type', 'radius')
    if input_type not in INPUT_REGISTRY:
        raise ValueError(f"Unknown input type: {input_type}. Available: {list(INPUT_REGISTRY.keys())}")

    # 检查 model 类型
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'schnet')
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")

    # 检查 output 类型
    output_configs = config.get('outputs', [])
    for out_config in output_configs:
        out_type = out_config.get('type')
        if out_type and out_type not in OUTPUT_REGISTRY:
            raise ValueError(f"Unknown output type: {out_type}. Available: {list(OUTPUT_REGISTRY.keys())}")

    return True


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    合并两个配置字典

    override 中的值会覆盖 base 中的值

    Args:
        base: 基础配置
        override: 覆盖配置

    Returns:
        合并后的配置
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


# 预定义配置模板
CONFIG_TEMPLATES = {
    'schnet-small': {
        'input': {'type': 'radius', 'cutoff': 5.0},
        'model': {'type': 'schnet', 'hidden_dim': 64, 'num_layers': 3},
        'outputs': [{'type': 'energy_and_forces'}],
    },
    'schnet-medium': {
        'input': {'type': 'radius', 'cutoff': 5.0},
        'model': {'type': 'schnet', 'hidden_dim': 128, 'num_layers': 6},
        'outputs': [{'type': 'energy_and_forces'}],
    },
    'schnet-large': {
        'input': {'type': 'radius', 'cutoff': 6.0},
        'model': {'type': 'schnet', 'hidden_dim': 256, 'num_layers': 6},
        'outputs': [{'type': 'energy_and_forces'}],
    },
    'tensornet-small': {
        'input': {'type': 'radius', 'cutoff': 5.0},
        'model': {'type': 'tensornet', 'hidden_dim': 64, 'num_layers': 1},
        'outputs': [{'type': 'energy_and_forces'}],
    },
    'tensornet-medium': {
        'input': {'type': 'radius', 'cutoff': 5.0},
        'model': {'type': 'tensornet', 'hidden_dim': 128, 'num_layers': 2},
        'outputs': [{'type': 'energy_and_forces'}],
    },
    'newtonnet-small': {
        'input': {'type': 'radius', 'cutoff': 5.0},
        'model': {'type': 'newtonnet', 'hidden_dim': 64, 'num_layers': 2},
        'outputs': [{'type': 'energy_and_forces'}],
    },
    'newtonnet-medium': {
        'input': {'type': 'radius', 'cutoff': 5.0},
        'model': {'type': 'newtonnet', 'hidden_dim': 128, 'num_layers': 3},
        'outputs': [{'type': 'energy_and_forces'}],
    },
}


def get_template(name: str) -> Dict[str, Any]:
    """
    获取预定义配置模板

    Args:
        name: 模板名称

    Returns:
        配置字典
    """
    if name not in CONFIG_TEMPLATES:
        raise ValueError(f"Unknown template: {name}. Available: {list(CONFIG_TEMPLATES.keys())}")

    # 返回深拷贝
    import copy
    return copy.deepcopy(CONFIG_TEMPLATES[name])


def list_templates() -> list:
    """列出所有可用的配置模板"""
    return list(CONFIG_TEMPLATES.keys())