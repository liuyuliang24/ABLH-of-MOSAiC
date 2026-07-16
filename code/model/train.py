# train.py
"""
训练函数模块
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import os

from loss_functions import CompositeLoss


def train_model_fixed(model, train_loader, val_loader, config):
    """Train model with overfitting fixes"""
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(device)

    # Loss function — 包含非对称惩罚 + 相对误差 + 偏差惩罚
    criterion = CompositeLoss(
        huber_delta=config.get('huber_delta', 1.0),
        mse_weight=config.get('mse_weight', 0.2),
        bias_weight=config.get('bias_weight', 0.15),
        asymmetry=config.get('asymmetry', 0.65),
        relative_weight=config.get('relative_weight', 0.1),
        condition_balance_weight=config.get('condition_balance_weight', 0.30),
        condition_bias_weight=config.get('condition_bias_weight', 0.45),
        spread_weight=config.get('spread_weight', 0.08),
        global_spread_weight=config.get('global_spread_weight', 0.0),
        global_slope_weight=config.get('global_slope_weight', 0.0),
        cloudy_focus_weight=config.get('cloudy_focus_weight', 0.14),
        cloudy_bias_weight=config.get('cloudy_bias_weight', 0.10),
        cloudy_high_threshold=config.get('cloudy_high_threshold', 320.0),
        cloudy_high_underestimate_weight=config.get('cloudy_high_underestimate_weight', 0.16),
        low_pblh_threshold=config.get('low_pblh_threshold', 250.0),
        low_pblh_weight=config.get('low_pblh_weight', 0.42),
        low_bias_weight=config.get('low_bias_weight', 0.18),
        low_overestimate_weight=config.get('low_overestimate_weight', 0.12),
        very_low_pblh_threshold=config.get('very_low_pblh_threshold', 120.0),
        very_low_bias_weight=config.get('very_low_bias_weight', 0.0),
        very_low_overestimate_weight=config.get('very_low_overestimate_weight', 0.0),
        high_pblh_threshold=config.get('high_pblh_threshold', 500.0),
        high_pblh_weight=config.get('high_pblh_weight', 0.0),
        high_bias_weight=config.get('high_bias_weight', 0.0),
        high_underestimate_weight=config.get('high_underestimate_weight', 0.0),
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 3e-4),
        weight_decay=config.get('weight_decay', 1e-3),
        betas=(0.9, 0.999)
    )

    # OneCycleLR: fast warmup (10%) + cosine decay
    epochs = config.get('epochs', 300)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.get('learning_rate', 3e-4),
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
        anneal_strategy='cos',
        final_div_factor=1e4,
    )

    # Mixed precision training
    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)

    # Early stopping
    best_val_loss = float('inf')
    best_val_r2 = -float('inf')
    patience = config.get('patience', 40)
    patience_counter = 0

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_rmse': [], 'val_rmse': [],
        'train_mae': [], 'val_mae': [],
        'train_r2': [], 'val_r2': [],
        'learning_rate': []
    }
    grad_clip = config.get('grad_clip', 1.0)

    for epoch in range(epochs):
        # ========== Training phase ==========
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')
        for batch in progress_bar:
            # 提取输入特征
            hatpro = batch['hatpro'].to(device)
            miracp = batch['miracp'].to(device)
            backscatter = batch['backscatter'].to(device)
            u_wind = batch['u_wind'].to(device)
            v_wind = batch['v_wind'].to(device)
            physics = batch['physics'].to(device)
            cloud_status = batch['cloud_status'].to(device)
            condition_label = batch['condition_label'].to(device)

            # 目标值
            target = batch['label'].to(device)
            raw_target = batch['raw_label'].to(device)
            weight = batch['condition_weight'].to(device)

            # 检查是否有红外数据
            aeri_rad = None
            if 'aeri_rad' in batch:
                aeri_rad = batch['aeri_rad'].to(device)

            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                # 根据是否有红外数据调用模型
                if aeri_rad is not None:
                    output = model(
                        hatpro=hatpro,
                        miracp=miracp,
                        backscatter=backscatter,
                        u_wind=u_wind,
                        v_wind=v_wind,
                        physics=physics,
                        cloud_status=cloud_status,
                        condition_label=condition_label,
                        aeri_rad=aeri_rad
                    )
                else:
                    # 向后兼容：没有红外数据时使用旧的调用方式
                    output = model(
                        hatpro=hatpro,
                        miracp=miracp,
                        backscatter=backscatter,
                        u_wind=u_wind,
                        v_wind=v_wind,
                        physics=physics,
                        cloud_status=cloud_status,
                        condition_label=condition_label
                    )

                loss = criterion(
                    output,
                    target,
                    weight=weight,
                    condition_label=condition_label,
                    raw_target=raw_target
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # OneCycleLR steps every batch
            scheduler.step()

            train_loss += loss.item() * target.size(0)
            train_preds.extend(output.detach().cpu().numpy())
            train_targets.extend(target.cpu().numpy())

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate training metrics
        train_loss /= len(train_loader.dataset)
        train_preds = np.array(train_preds).flatten()
        train_targets = np.array(train_targets).flatten()
        train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
        train_mae = mean_absolute_error(train_targets, train_preds)
        train_r2 = r2_score(train_targets, train_preds) if len(train_preds) > 1 else 0

        # ========== Validation phase ==========
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]'):
                # 提取输入特征
                hatpro = batch['hatpro'].to(device)
                miracp = batch['miracp'].to(device)
                backscatter = batch['backscatter'].to(device)
                u_wind = batch['u_wind'].to(device)
                v_wind = batch['v_wind'].to(device)
                physics = batch['physics'].to(device)
                cloud_status = batch['cloud_status'].to(device)
                condition_label = batch['condition_label'].to(device)

                # 目标值
                target = batch['label'].to(device)
                raw_target = batch['raw_label'].to(device)

                # 检查是否有红外数据
                aeri_rad = None
                if 'aeri_rad' in batch:
                    aeri_rad = batch['aeri_rad'].to(device)

                with autocast(enabled=use_amp):
                    # 根据是否有红外数据调用模型
                    if aeri_rad is not None:
                        output = model(
                            hatpro=hatpro,
                            miracp=miracp,
                            backscatter=backscatter,
                            u_wind=u_wind,
                            v_wind=v_wind,
                            physics=physics,
                            cloud_status=cloud_status,
                            condition_label=condition_label,
                            aeri_rad=aeri_rad
                        )
                    else:
                        # 向后兼容
                        output = model(
                            hatpro=hatpro,
                            miracp=miracp,
                            backscatter=backscatter,
                            u_wind=u_wind,
                            v_wind=v_wind,
                            physics=physics,
                            cloud_status=cloud_status,
                            condition_label=condition_label
                        )

                    loss = criterion(
                        output,
                        target,
                        condition_label=condition_label,
                        raw_target=raw_target
                    )

                val_loss += loss.item() * target.size(0)
                val_preds.extend(output.cpu().numpy())
                val_targets.extend(target.cpu().numpy())

        # Calculate validation metrics
        val_loss /= len(val_loader.dataset)
        val_preds = np.array(val_preds).flatten()
        val_targets = np.array(val_targets).flatten()
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_r2 = r2_score(val_targets, val_preds) if len(val_preds) > 1 else 0

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # Save best model — 以 val_loss 为主，同时记录最佳 R²
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_r2 = val_r2
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'config': config
            }

            model_path = os.path.join(config['output_dir'], config['model_save_path'])
            torch.save(checkpoint, model_path)

            print(f'\nEpoch {epoch + 1}: ✓ Saved best model '
                  f'(Val Loss: {val_loss:.6f}, Val RMSE: {val_rmse:.2f}, Val R²: {val_r2:.3f})')
        else:
            patience_counter += 1

        # Print progress every epoch
        print(f'\nEpoch {epoch + 1}/{epochs}:')
        print(f'  Train - Loss: {train_loss:.4f}, RMSE: {train_rmse:.4f}, '
              f'MAE: {train_mae:.4f}, R²: {train_r2:.3f}')
        print(f'  Val   - Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f}, '
              f'MAE: {val_mae:.4f}, R²: {val_r2:.3f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}, '
              f'Patience: {patience_counter}/{patience}')

        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping at Epoch {epoch + 1} '
                  f'(best Val Loss: {best_val_loss:.6f}, best Val R²: {best_val_r2:.3f})')
            break

    return model, history


def create_training_config(output_dir='./results', model_name='pbl_model', **kwargs):
    """
    创建训练配置

    参数
    ----------
    output_dir : str
        输出目录
    model_name : str
        模型保存名称
    **kwargs : dict
        其他配置参数

    返回
    -------
    dict
        训练配置字典
    """
    # 默认配置
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'learning_rate': 2e-4,
        'weight_decay': 6e-4,
        'epochs': 320,
        'patience': 40,
        'batch_size': 48,
        'grad_clip': 1.0,

        # 损失函数参数
        'huber_delta': 1.0,
        'mse_weight': 0.2,
        'bias_weight': 0.18,
        'asymmetry': 0.65,
        'relative_weight': 0.12,
        'condition_balance_weight': 0.30,
        'condition_bias_weight': 0.45,
        'spread_weight': 0.08,
        'global_spread_weight': 0.10,
        'global_slope_weight': 0.14,
        'cloudy_focus_weight': 0.14,
        'cloudy_bias_weight': 0.10,
        'cloudy_high_threshold': 320.0,
        'cloudy_high_underestimate_weight': 0.16,
        'low_pblh_threshold': 250.0,
        'low_pblh_weight': 0.42,
        'low_bias_weight': 0.18,
        'low_overestimate_weight': 0.12,
        'very_low_pblh_threshold': 120.0,
        'very_low_bias_weight': 0.10,
        'very_low_overestimate_weight': 0.20,
        'high_pblh_threshold': 500.0,
        'high_pblh_weight': 0.24,
        'high_bias_weight': 0.12,
        'high_underestimate_weight': 0.30,
        'cloudy_condition_boost': 1.15,
        'cloudy_low_boost': 1.08,
        'cloudy_mid_boost': 1.12,
        'cloudy_high_boost': 1.18,

        # 输出路径
        'output_dir': output_dir,
        'model_save_path': f'{model_name}_best.pth',
        'history_save_path': f'{model_name}_history.pkl',
        'config_save_path': f'{model_name}_config.yaml',

        # 模型参数
        'hatpro_dim': 14,
        'miracp_dim': 8,
        'ceil_dim': 200,
        'wind_dim': 100,
        'use_wind_branch': True,
        'physics_dim': None,
        'dropout_rate': 0.18,
        'n_cloud_classes': 5,
        'n_condition_classes': 3,
        'near_surface_bins': 48,
        'condition_prior_scale': 1.5,
        'cloudy_refine_scale': 0.24,
        'hatpro_k_band_dim': 7,
        'hatpro_v_band_dim': 7,
        'miracp_absorption_dim': 6,
        'miracp_window_dim': 2,
        'use_wind_features': True,
    }

    # 更新用户提供的参数
    config.update(kwargs)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    return config
