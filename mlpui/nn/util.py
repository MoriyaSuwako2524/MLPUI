"""
mlpui/nn/utils.py

工具函数
========
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch import Tensor


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    统计模型参数数量

    Args:
        model: PyTorch 模型
        trainable_only: 是否只统计可训练参数

    Returns:
        参数数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_parameters(num_params: int) -> str:
    """
    格式化参数数量

    Args:
        num_params: 参数数量

    Returns:
        格式化字符串 (如 "1.2M", "500K")
    """
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


def model_summary(model: nn.Module, depth: int = 2) -> str:
    """
    生成模型摘要

    Args:
        model: PyTorch 模型
        depth: 显示深度

    Returns:
        摘要字符串
    """
    lines = []

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    lines.append(f"{'=' * 60}")
    lines.append(f"Model Summary")
    lines.append(f"{'=' * 60}")

    # 递归打印模块
    def print_module(module, name, indent):
        if indent // 2 >= depth:
            return

        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            lines.append(f"{' ' * indent}{name}: {format_parameters(params)}")
        else:
            lines.append(f"{' ' * indent}{name}")

        for child_name, child in module.named_children():
            print_module(child, child_name, indent + 2)

    for name, module in model.named_children():
        print_module(module, name, 0)

    lines.append(f"{'=' * 60}")
    lines.append(f"Total params: {format_parameters(total_params)}")
    lines.append(f"Trainable params: {format_parameters(trainable_params)}")
    lines.append(f"Non-trainable params: {format_parameters(total_params - trainable_params)}")
    lines.append(f"{'=' * 60}")

    return '\n'.join(lines)


def get_optimizer(
        model: nn.Module,
        optimizer: str = 'adam',
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        **kwargs,
) -> torch.optim.Optimizer:
    """
    获取优化器

    Args:
        model: 模型
        optimizer: 优化器名称
        lr: 学习率
        weight_decay: 权重衰减
        **kwargs: 其他参数

    Returns:
        优化器实例
    """
    params = filter(lambda p: p.requires_grad, model.parameters())

    optimizers = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'rmsprop': torch.optim.RMSprop,
    }

    if optimizer.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer}. Available: {list(optimizers.keys())}")

    opt_class = optimizers[optimizer.lower()]
    return opt_class(params, lr=lr, weight_decay=weight_decay, **kwargs)


def get_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler: str = 'plateau',
        **kwargs,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    获取学习率调度器

    Args:
        optimizer: 优化器
        scheduler: 调度器名称
        **kwargs: 其他参数

    Returns:
        调度器实例
    """
    if scheduler is None or scheduler == 'none':
        return None

    schedulers = {
        'plateau': lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            min_lr=kwargs.get('min_lr', 1e-7),
        ),
        'cosine': lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 1e-7),
        ),
        'step': lambda: torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1),
        ),
        'exponential': lambda: torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.99),
        ),
    }

    if scheduler.lower() not in schedulers:
        raise ValueError(f"Unknown scheduler: {scheduler}. Available: {list(schedulers.keys())}")

    return schedulers[scheduler.lower()]()


def apply_ema(
        model: nn.Module,
        ema_model: nn.Module,
        decay: float = 0.999,
):
    """
    应用指数移动平均 (EMA)

    Args:
        model: 当前模型
        ema_model: EMA 模型
        decay: 衰减率
    """
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


class EMAModel:
    """
    指数移动平均模型包装器

    Example:
        >>> ema = EMAModel(model, decay=0.999)
        >>> # 训练循环中
        >>> ema.update()
        >>> # 评估时
        >>> with ema.average_parameters():
        ...     evaluate(model)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化影子参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新 EMA 参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        """应用 EMA 参数到模型"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def average_parameters(self):
        """上下文管理器: 临时使用 EMA 参数"""

        class _Context:
            def __init__(ctx):
                pass

            def __enter__(ctx):
                self.apply_shadow()
                return self.model

            def __exit__(ctx, *args):
                self.restore()

        return _Context()


def compute_stats(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cpu',
) -> Dict[str, float]:
    """
    计算数据集上的统计信息

    Args:
        model: UnifiedMLP 模型
        dataloader: 数据加载器
        device: 设备

    Returns:
        统计字典 (mean, std 等)
    """
    model = model.to(device)
    model.eval()

    all_energies = []
    all_forces = []

    with torch.no_grad():
        for batch in dataloader:
            # 移动到设备
            if hasattr(batch, 'to'):
                batch = batch.to(device)

            results = model(batch, compute_forces=False)

            if 'energy' in results:
                all_energies.append(results['energy'].cpu())

    stats = {}

    if all_energies:
        energies = torch.cat(all_energies)
        stats['energy_mean'] = energies.mean().item()
        stats['energy_std'] = energies.std().item()

    return stats