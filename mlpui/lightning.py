"""
MLPUI - PyTorch Lightning Module
================================

用于训练的 PyTorch Lightning 接口
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from mlpui.model import UnifiedMLP, create_unified_model


class MLPLitModule(pl.LightningModule):
    """
    PyTorch Lightning 训练模块

    Features:
    - 多属性损失计算
    - 自动力计算
    - 灵活的损失权重
    - 学习率调度
    - EMA (可选)

    Example:
        >>> module = MLPLitModule(
        ...     mlp='tensornet',
        ...     properties=['energy', 'forces', 'dipole'],
        ...     loss_weights={'energy': 1, 'forces': 100, 'dipole': 10},
        ...     hidden_channels=128,
        ... )
        >>> trainer = pl.Trainer(max_epochs=100)
        >>> trainer.fit(module, train_loader, val_loader)
    """

    def __init__(
            self,
            mlp: str = 'tensornet',
            properties: List[str] = ['energy', 'forces'],
            loss_weights: Optional[Dict[str, float]] = None,
            loss_type: str = 'mse',
            # 优化器
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            lr_scheduler: str = 'reduce_on_plateau',
            lr_patience: int = 10,
            lr_factor: float = 0.5,
            lr_min: float = 1e-6,
            # EMA
            ema_decay: Optional[float] = None,
            # 头配置
            head_configs: Optional[Dict[str, Dict]] = None,
            # MLP 配置
            **mlp_kwargs
    ):
        """
        初始化

        Args:
            mlp: MLP 类型
            properties: 属性列表
            loss_weights: 损失权重
            loss_type: 损失类型 ('mse', 'mae', 'huber')
            learning_rate: 学习率
            weight_decay: 权重衰减
            lr_scheduler: 学习率调度器
            head_configs: 输出头配置
            **mlp_kwargs: MLP 参数
        """
        super().__init__()

        self.save_hyperparameters()

        # 创建模型
        self.model = create_unified_model(
            mlp=mlp,
            properties=properties,
            head_configs=head_configs,
            **mlp_kwargs
        )

        self.properties = properties
        self.loss_weights = loss_weights or {p: 1.0 for p in properties}

        # 损失函数
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'mae':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # EMA
        self.ema_decay = ema_decay
        if ema_decay:
            import copy
            self.ema_model = copy.deepcopy(self.model)
            for p in self.ema_model.parameters():
                p.requires_grad = False

    def _update_ema(self):
        """更新 EMA 参数"""
        if self.ema_decay is None:
            return
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def forward(self, z, pos, batch, cell=None, **kwargs):
        """前向传播"""
        return self.model(z, pos, batch, cell, **kwargs)

    def _extract_batch(self, batch) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Dict[str, Tensor]]:
        """从批次提取数据"""
        if hasattr(batch, 'z'):
            # torch_geometric 格式
            z = batch.z
            pos = batch.pos
            batch_idx = batch.batch
            cell = getattr(batch, 'cell', None)

            targets = {}
            for prop in self.properties:
                if hasattr(batch, prop):
                    targets[prop] = getattr(batch, prop)
                elif prop == 'forces' and hasattr(batch, 'neg_dy'):
                    targets['forces'] = batch.neg_dy
                elif prop == 'energy' and hasattr(batch, 'y'):
                    targets['energy'] = batch.y
        else:
            # 字典格式
            z = batch['z']
            pos = batch['pos']
            batch_idx = batch['batch']
            cell = batch.get('cell')
            targets = {k: v for k, v in batch.items() if k in self.properties}

        return z, pos, batch_idx, cell, targets

    def _compute_loss(
            self,
            preds: Dict[str, Tensor],
            targets: Dict[str, Tensor],
            prefix: str = ''
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """计算损失"""
        total_loss = torch.tensor(0.0, device=self.device)
        losses = {}

        for prop in self.properties:
            if prop not in preds or prop not in targets:
                continue

            pred = preds[prop]
            target = targets[prop]

            # 确保形状匹配
            if pred.shape != target.shape:
                if pred.dim() == 1 and target.dim() == 2:
                    pred = pred.unsqueeze(-1)
                elif pred.dim() == 2 and target.dim() == 1:
                    target = target.unsqueeze(-1)

            loss = self.loss_fn(pred, target)
            weight = self.loss_weights.get(prop, 1.0)

            total_loss = total_loss + weight * loss
            losses[f'{prefix}{prop}_loss'] = loss.detach()

            # MAE
            mae = (pred - target).abs().mean()
            losses[f'{prefix}{prop}_mae'] = mae.detach()

        losses[f'{prefix}total_loss'] = total_loss.detach()

        return total_loss, losses

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        z, pos, batch_idx_t, cell, targets = self._extract_batch(batch)

        preds = self.model(z, pos, batch_idx_t, cell)
        loss, losses = self._compute_loss(preds, targets, 'train_')

        self.log_dict(losses, on_step=True, on_epoch=True, prog_bar=True)

        if self.ema_decay:
            self._update_ema()

        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        z, pos, batch_idx_t, cell, targets = self._extract_batch(batch)

        model = self.ema_model if self.ema_decay else self.model

        with torch.no_grad():
            preds = model(z, pos, batch_idx_t, cell)

        loss, losses = self._compute_loss(preds, targets, 'val_')
        self.log_dict(losses, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """测试步骤"""
        z, pos, batch_idx_t, cell, targets = self._extract_batch(batch)

        model = self.ema_model if self.ema_decay else self.model

        with torch.no_grad():
            preds = model(z, pos, batch_idx_t, cell)

        loss, losses = self._compute_loss(preds, targets, 'test_')
        self.log_dict(losses, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """配置优化器"""
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.lr_scheduler == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.hparams.lr_factor,
                patience=self.hparams.lr_patience,
                min_lr=self.hparams.lr_min,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_total_loss',
                }
            }
        elif self.hparams.lr_scheduler == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.hparams.lr_min,
            )
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

        return optimizer


class MLPDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning 数据模块
    """

    def __init__(
            self,
            train_data=None,
            val_data=None,
            test_data=None,
            batch_size: int = 32,
            num_workers: int = 4,
            pin_memory: bool = True,
    ):
        super().__init__()

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.test_data is None:
            return None
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )