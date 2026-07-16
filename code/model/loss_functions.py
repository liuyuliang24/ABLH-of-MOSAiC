# loss_functions.py
"""
损失函数模块
"""
import torch
import torch.nn as nn


class CompositeLoss(nn.Module):
    """
    Composite loss: AsymmetricHuber + MSE + bias-penalty.

    改进点：
    1. 非对称 Huber：对预测值低于真值（低估）施加更大惩罚，
       解决散点图中高PBLH被系统性低估的问题。
    2. 保留偏差惩罚项，防止系统性偏移。
    3. 增加相对误差项，对低值区和高值区均衡惩罚。
    4. 增加动态范围约束，缓解“低层偏大、高层偏低”的斜率压缩。
    """

    def __init__(self,
                 huber_delta=1.0,
                 mse_weight=0.3,
                 bias_weight=0.15,
                 asymmetry=0.6,
                 relative_weight=0.1,
                 condition_balance_weight=0.35,
                 condition_bias_weight=0.4,
                 spread_weight=0.06,
                 cloudy_focus_weight=0.0,
                 cloudy_bias_weight=0.0,
                 cloudy_high_threshold=320.0,
                 cloudy_high_underestimate_weight=0.0,
                 low_pblh_threshold=250.0,
                 low_pblh_weight=0.22,
                 low_bias_weight=0.18,
                 low_overestimate_weight=0.12,
                 very_low_pblh_threshold=120.0,
                 very_low_bias_weight=0.0,
                 very_low_overestimate_weight=0.0,
                 high_pblh_threshold=500.0,
                 high_pblh_weight=0.0,
                 high_bias_weight=0.0,
                 high_underestimate_weight=0.0,
                 global_spread_weight=0.0,
                 global_slope_weight=0.0):
        """
        Parameters
        ----------
        huber_delta : float
            Huber loss 的 delta 参数（在标准化空间中）。
        mse_weight : float
            MSE 项的权重。
        bias_weight : float
            偏差惩罚项的权重（防止系统性偏移）。
        asymmetry : float
            非对称系数 ∈ (0.5, 1.0)。
            > 0.5 表示低估（pred < target）受到更大惩罚。
            0.5 = 对称（等价于标准 Huber）。
        relative_weight : float
            相对误差项权重，使得高值和低值区域误差均衡。
        """
        super().__init__()
        self.huber = nn.HuberLoss(delta=huber_delta, reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        self.huber_delta = huber_delta
        self.mse_weight = mse_weight
        self.bias_weight = bias_weight
        self.asymmetry = asymmetry  # 低估惩罚系数
        self.relative_weight = relative_weight
        self.condition_balance_weight = condition_balance_weight
        self.condition_bias_weight = condition_bias_weight
        self.spread_weight = spread_weight
        self.cloudy_focus_weight = cloudy_focus_weight
        self.cloudy_bias_weight = cloudy_bias_weight
        self.cloudy_high_threshold = cloudy_high_threshold
        self.cloudy_high_underestimate_weight = cloudy_high_underestimate_weight
        self.low_pblh_threshold = low_pblh_threshold
        self.low_pblh_weight = low_pblh_weight
        self.low_bias_weight = low_bias_weight
        self.low_overestimate_weight = low_overestimate_weight
        self.very_low_pblh_threshold = very_low_pblh_threshold
        self.very_low_bias_weight = very_low_bias_weight
        self.very_low_overestimate_weight = very_low_overestimate_weight
        self.high_pblh_threshold = high_pblh_threshold
        self.high_pblh_weight = high_pblh_weight
        self.high_bias_weight = high_bias_weight
        self.high_underestimate_weight = high_underestimate_weight
        self.global_spread_weight = global_spread_weight
        self.global_slope_weight = global_slope_weight

    def forward(self, pred, target, weight=None, condition_label=None, raw_target=None):
        # ── 基础 Huber 损失（element-wise）──────────────────────────────
        huber_loss = self.huber(pred, target)  # (B, 1)
        mse_loss = self.mse(pred, target)  # (B, 1)

        # ── 非对称加权 ────────────────────────────────────────────────
        # residual > 0 → 低估（pred < target），给更大权重
        # residual < 0 → 高估（pred > target），给较小权重
        residual = target - pred  # (B, 1)
        # 低估: asymmetry (>0.5)，高估: 1 - asymmetry (<0.5)
        asym_weight = torch.where(
            residual > 0,
            torch.full_like(residual, self.asymmetry),
            torch.full_like(residual, 1.0 - self.asymmetry)
        )
        # 归一化到均值1，避免改变损失量级
        asym_weight = asym_weight / asym_weight.mean().clamp(min=1e-8)

        huber_loss = huber_loss * asym_weight
        mse_loss = mse_loss * asym_weight

        # ── 相对误差项（MAPE-like，但更稳定）────────────────────────────
        # 在标准化空间中 target 有时接近0，用 |target|+eps 保护
        rel_loss = (residual ** 2) / (target.abs() + 0.1) ** 2

        pointwise_loss = (
            huber_loss
            + self.mse_weight * mse_loss
            + self.relative_weight * rel_loss
        )

        # ── 样本权重（来自数据集的 condition_weight）────────────────────
        if weight is not None:
            w = weight.view(-1, 1) if weight.dim() == 1 else weight
            w = w / (w.mean() + 1e-8)
            pointwise_loss = pointwise_loss * w

        base_loss = pointwise_loss.mean()

        # ── 偏差惩罚（防止系统性偏移）────────────────────────────────
        global_bias_loss = (pred - target).mean() ** 2

        condition_bias_loss = pred.new_tensor(0.0)
        condition_spread_loss = pred.new_tensor(0.0)
        cloudy_focus_loss = pred.new_tensor(0.0)
        cloudy_bias_loss = pred.new_tensor(0.0)
        cloudy_high_underestimate_loss = pred.new_tensor(0.0)
        low_pblh_loss = pred.new_tensor(0.0)
        low_bias_loss = pred.new_tensor(0.0)
        low_overestimate_loss = pred.new_tensor(0.0)
        very_low_bias_loss = pred.new_tensor(0.0)
        very_low_overestimate_loss = pred.new_tensor(0.0)
        high_pblh_loss = pred.new_tensor(0.0)
        high_bias_loss = pred.new_tensor(0.0)
        high_underestimate_loss = pred.new_tensor(0.0)

        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        target_var = target_centered.pow(2).mean().clamp(min=1e-6)
        pred_std = pred_centered.pow(2).mean().sqrt()
        target_std = target_var.sqrt()
        global_spread_loss = torch.relu(target_std - pred_std).pow(2)
        global_slope = (pred_centered * target_centered).mean() / target_var
        global_slope_loss = torch.relu(1.0 - global_slope).pow(2)

        # 对不同天气条件做平衡，避免多云/晴空等大样本条件压制其它条件
        if condition_label is not None:
            condition_label = condition_label.view(-1).long()
            grouped_losses = []
            grouped_biases = []
            grouped_spreads = []

            for cond in torch.unique(condition_label):
                mask = condition_label == cond
                if not torch.any(mask):
                    continue

                cond_pointwise = pointwise_loss[mask]
                cond_pred = pred[mask]
                cond_target = target[mask]

                grouped_losses.append(cond_pointwise.mean())
                grouped_biases.append((cond_pred.mean() - cond_target.mean()) ** 2)

                if cond_pred.numel() > 1:
                    grouped_spreads.append(
                        (cond_pred.std(unbiased=False) - cond_target.std(unbiased=False)) ** 2
                    )

            if grouped_losses:
                balanced_loss = torch.stack(grouped_losses).mean()
                base_loss = (
                    (1.0 - self.condition_balance_weight) * base_loss
                    + self.condition_balance_weight * balanced_loss
                )
            if grouped_biases:
                condition_bias_loss = torch.stack(grouped_biases).mean()
            if grouped_spreads:
                condition_spread_loss = torch.stack(grouped_spreads).mean()

            cloudy_mask = condition_label == 1
            if torch.any(cloudy_mask):
                cloudy_pred = pred[cloudy_mask]
                cloudy_target = target[cloudy_mask]
                cloudy_focus_loss = pointwise_loss[cloudy_mask].mean()
                cloudy_bias_loss = (cloudy_pred.mean() - cloudy_target.mean()) ** 2

                if raw_target is not None:
                    raw_target_view = raw_target.view(-1)
                    cloudy_high_mask = cloudy_mask & (raw_target_view >= self.cloudy_high_threshold)
                    if torch.any(cloudy_high_mask):
                        cloudy_high_residual = target[cloudy_high_mask] - pred[cloudy_high_mask]
                        cloudy_high_underestimate_loss = torch.relu(cloudy_high_residual).pow(2).mean()

        # 低边界层高度样本单独增强，缓解浅边界层被均值化的问题
        if raw_target is not None:
            raw_target = raw_target.view_as(target)
            low_mask = raw_target <= self.low_pblh_threshold
            very_low_mask = raw_target <= self.very_low_pblh_threshold
            high_mask = raw_target >= self.high_pblh_threshold

            if torch.any(low_mask):
                low_pointwise = pointwise_loss[low_mask]
                low_pred = pred[low_mask]
                low_target = target[low_mask]
                low_residual = low_pred - low_target

                low_pblh_loss = low_pointwise.mean()
                low_bias_loss = low_residual.mean() ** 2
                low_overestimate_loss = torch.relu(low_residual).pow(2).mean()

            if torch.any(very_low_mask):
                very_low_pred = pred[very_low_mask]
                very_low_target = target[very_low_mask]
                very_low_residual = very_low_pred - very_low_target

                very_low_bias_loss = very_low_residual.mean() ** 2
                very_low_overestimate_loss = torch.relu(very_low_residual).pow(2).mean()

            if torch.any(high_mask):
                high_pointwise = pointwise_loss[high_mask]
                high_pred = pred[high_mask]
                high_target = target[high_mask]
                high_residual = high_target - high_pred

                high_pblh_loss = high_pointwise.mean()
                high_bias_loss = (high_pred.mean() - high_target.mean()) ** 2
                high_underestimate_loss = torch.relu(high_residual).pow(2).mean()

        total = (
            base_loss
            + self.bias_weight * (
                global_bias_loss
                + self.condition_bias_weight * condition_bias_loss
            )
            + self.spread_weight * condition_spread_loss
            + self.global_spread_weight * global_spread_loss
            + self.global_slope_weight * global_slope_loss
            + self.cloudy_focus_weight * cloudy_focus_loss
            + self.cloudy_bias_weight * cloudy_bias_loss
            + self.cloudy_high_underestimate_weight * cloudy_high_underestimate_loss
            + self.low_pblh_weight * low_pblh_loss
            + self.low_bias_weight * low_bias_loss
            + self.low_overestimate_weight * low_overestimate_loss
            + self.very_low_bias_weight * very_low_bias_loss
            + self.very_low_overestimate_weight * very_low_overestimate_loss
            + self.high_pblh_weight * high_pblh_loss
            + self.high_bias_weight * high_bias_loss
            + self.high_underestimate_weight * high_underestimate_loss
        )
        return total


# 其他常用的损失函数
class ConditionWeightedLoss(nn.Module):
    """
    根据天气条件加权的损失函数

    不同天气条件（晴天/多云/雾天）对模型难度不同，
    给困难条件更高权重
    """

    def __init__(self, base_loss='huber', condition_weights=None):
        """
        Parameters
        ----------
        base_loss : str
            基础损失函数，可选 'huber', 'mse', 'l1'
        condition_weights : dict
            天气条件权重，格式 {condition_label: weight}
        """
        super().__init__()

        if base_loss == 'huber':
            self.base_criterion = nn.HuberLoss()
        elif base_loss == 'mse':
            self.base_criterion = nn.MSELoss()
        elif base_loss == 'l1':
            self.base_criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown base_loss: {base_loss}")

        # 默认条件权重
        if condition_weights is None:
            condition_weights = {0: 1.0, 1: 1.5, 2: 1.2}  # 晴/阴/雾
        self.condition_weights = condition_weights

    def forward(self, pred, target, condition_labels):
        """
        参数
        ----------
        pred : torch.Tensor
            预测值
        target : torch.Tensor
            真实值
        condition_labels : torch.Tensor
            天气条件标签，0: 晴, 1: 阴, 2: 雾
        """
        # 计算基础损失
        base_loss = self.base_criterion(pred, target)

        # 为每个样本计算权重
        weights = torch.ones_like(condition_labels, dtype=torch.float32)
        for cond, weight in self.condition_weights.items():
            mask = condition_labels == cond
            weights[mask] = weight

        # 权重归一化
        weights = weights / weights.mean()

        # 加权损失
        weighted_loss = (base_loss.unsqueeze(-1) * weights).mean()

        return weighted_loss


class PBLHAsymmetricLoss(nn.Module):
    """
    专门针对PBLH估计的非对称损失函数

    特点：
    1. 对高PBLH值的低估施加更强惩罚
    2. 对低PBLH值的高估施加适当惩罚
    3. 包含边界约束，防止极端预测
    """

    def __init__(self, high_pblh_threshold=500.0, asymmetry_factor=2.0):
        super().__init__()
        self.high_threshold = high_pblh_threshold
        self.asymmetry_factor = asymmetry_factor

    def forward(self, pred, target):
        """
        参数
        ----------
        pred : torch.Tensor
            预测的PBLH值
        target : torch.Tensor
            真实的PBLH值
        """
        residual = target - pred

        # 高PBLH（> threshold）且被低估的情况
        high_mask = (target > self.high_threshold) & (residual > 0)

        # 低PBLH（< threshold）且被高估的情况
        low_mask = (target <= self.high_threshold) & (residual < 0)

        # 基础损失
        base_loss = torch.abs(residual)

        # 应用非对称惩罚
        loss = base_loss.clone()
        loss[high_mask] *= self.asymmetry_factor
        loss[low_mask] *= (2.0 - self.asymmetry_factor)

        # 边界约束：防止负值或极大值
        boundary_penalty = torch.relu(-pred) + torch.relu(pred - 3000)  # 假设3000m为物理上限

        return loss.mean() + 0.1 * boundary_penalty.mean()


# 多任务损失函数
class MultiTaskLoss(nn.Module):
    """
    多任务学习损失函数

    同时优化：
    1. PBLH回归任务
    2. 天气条件分类任务（可选）
    3. 不确定性估计（可选）
    """

    def __init__(self, pblh_loss_weight=1.0,
                 condition_loss_weight=0.1,
                 uncertainty_weight=0.05):
        super().__init__()
        self.pblh_loss_weight = pblh_loss_weight
        self.condition_loss_weight = condition_loss_weight
        self.uncertainty_weight = uncertainty_weight

        # 回归损失
        self.regression_loss = CompositeLoss()

        # 分类损失
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, pblh_pred, pblh_target,
                condition_pred=None, condition_target=None,
                uncertainty=None):
        """
        参数
        ----------
        pblh_pred : torch.Tensor
            PBLH预测值
        pblh_target : torch.Tensor
            PBLH真实值
        condition_pred : torch.Tensor, optional
            天气条件预测概率
        condition_target : torch.Tensor, optional
            天气条件真实标签
        uncertainty : torch.Tensor, optional
            不确定性估计
        """
        # 回归损失
        reg_loss = self.regression_loss(pblh_pred, pblh_target)
        total_loss = self.pblh_loss_weight * reg_loss

        # 分类损失
        if condition_pred is not None and condition_target is not None:
            cls_loss = self.classification_loss(condition_pred, condition_target)
            total_loss += self.condition_loss_weight * cls_loss

        # 不确定性正则化
        if uncertainty is not None:
            # 鼓励适中的不确定性，避免过度自信或不自信
            uncertainty_reg = torch.abs(uncertainty.mean() - 0.5)
            total_loss += self.uncertainty_weight * uncertainty_reg

        return total_loss


# 工具函数
def get_loss_function(loss_name='composite', **kwargs):
    """
    获取损失函数工厂函数

    参数
    ----------
    loss_name : str
        损失函数名称，可选：
        - 'composite': 复合损失（默认）
        - 'condition_weighted': 条件加权损失
        - 'pblh_asymmetric': PBLH非对称损失
        - 'multi_task': 多任务损失
        - 'mse': 均方误差
        - 'huber': Huber损失
        - 'l1': L1损失
    **kwargs : dict
        损失函数参数

    返回
    -------
    nn.Module
        损失函数实例
    """
    if loss_name == 'composite':
        return CompositeLoss(**kwargs)
    elif loss_name == 'condition_weighted':
        return ConditionWeightedLoss(**kwargs)
    elif loss_name == 'pblh_asymmetric':
        return PBLHAsymmetricLoss(**kwargs)
    elif loss_name == 'multi_task':
        return MultiTaskLoss(**kwargs)
    elif loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'huber':
        delta = kwargs.get('delta', 1.0)
        return nn.HuberLoss(delta=delta)
    elif loss_name == 'l1':
        return nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
