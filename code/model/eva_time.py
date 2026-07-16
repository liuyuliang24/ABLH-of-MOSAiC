# evaluate_main.py
"""
评估主程序 - 评估训练集和验证集
"""
import os
import sys

import traceback
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

matplotlib.use('Agg')  # 设置为非交互式后端，避免GUI问题

# 导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TRAIN_CONFIG, MODEL_CONFIG
from utils import create_output_dir, get_device, safe_load_checkpoint
from dataset import PBLDatasetFixed
from models import create_pbl_model
from plots import plot_enhanced_scatter_plots, plot_evaluation_results, plot_training_history


def load_training_scalers():
    """加载训练阶段保存的 scaler，评估必须复用训练尺度。"""
    scaler_path = os.path.join(TRAIN_CONFIG['output_dir'], TRAIN_CONFIG['scaler_save_path'])
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Training scaler file not found: {scaler_path}. "
            f"Please run train_main.py successfully before evaluation."
        )
    print(f"Loading training scalers from: {scaler_path}")
    return joblib.load(scaler_path)


def load_train_val_datasets(external_scalers):
    """加载训练集和验证集"""
    print(f"\nLoading train and validation datasets...")

    datasets = {}
    data_loaders = {}
    use_aeri = TRAIN_CONFIG.get('use_aeri', True)
    use_microwave = TRAIN_CONFIG.get('use_microwave', True)
    use_wind_features = TRAIN_CONFIG.get('use_wind_features', True)
    use_ceilometer = TRAIN_CONFIG.get('use_ceilometer', True)
    aeri_channels = TRAIN_CONFIG.get('aeri_channels', 0) if use_aeri else 0

    for mode in ['train', 'val']:
        print(f"\nLoading {mode} dataset...")
        try:
            dataset = PBLDatasetFixed(
                features_file=TRAIN_CONFIG['features_file1'],
                labels_file=TRAIN_CONFIG['labels_file1'],
                mode=mode,
                split_method=TRAIN_CONFIG['split_method'],
                train_ratio=TRAIN_CONFIG['train_ratio'],
                val_ratio=TRAIN_CONFIG['val_ratio'],
                normalize=True,
                use_robust_scaler=TRAIN_CONFIG['use_robust_scaler'],
                laser_weight=TRAIN_CONFIG['laser_weight'],
                wind_weight=TRAIN_CONFIG['wind_weight'],
                use_wind_features=use_wind_features,
                max_samples=TRAIN_CONFIG['max_samples'],
                external_scalers=external_scalers,
                cloudy_condition_boost=TRAIN_CONFIG.get('cloudy_condition_boost', 1.0),
                cloudy_low_boost=TRAIN_CONFIG.get('cloudy_low_boost', 1.0),
                cloudy_mid_boost=TRAIN_CONFIG.get('cloudy_mid_boost', 1.0),
                cloudy_high_boost=TRAIN_CONFIG.get('cloudy_high_boost', 1.0),
                aeri_channels=aeri_channels,
                use_aeri=use_aeri,
                use_microwave=use_microwave,
                use_ceilometer=use_ceilometer,
            )

            datasets[mode] = dataset

            # 创建数据加载器
            data_loader = DataLoader(
                dataset,
                batch_size=TRAIN_CONFIG['batch_size'],
                shuffle=False,
                num_workers=0
            )
            data_loaders[mode] = data_loader

            print(f"  {mode} set: {len(dataset)} samples")

        except Exception as e:
            print(f"Error loading {mode} dataset: {e}")
            traceback.print_exc()
            return None, None

    return datasets, data_loaders


def predict_train_val_datasets(model, train_loader, val_loader, train_dataset, val_dataset, device):
    """对训练集和验证集进行预测，包含时间信息"""
    model = model.to(device)
    model.eval()

    # 从数据集中直接获取所有时间
    train_times = train_dataset.get_all_times() if hasattr(train_dataset, 'get_all_times') else None
    val_times = val_dataset.get_all_times() if hasattr(val_dataset, 'get_all_times') else None

    # 预测训练集
    train_preds, train_targets, train_cloud_status = [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_loader, desc='Predicting training set')):
            inputs = {k: v.to(device) for k, v in batch.items()
                      if k not in ['condition_weight', 'label', 'raw_label']}
            target = batch['label'].to(device)
            cloud_status = batch['cloud_status'].to(device)

            output = model(**inputs)

            train_preds.extend(output.cpu().numpy())
            train_targets.extend(target.cpu().numpy())
            train_cloud_status.extend(cloud_status.cpu().numpy())

    # 预测验证集
    val_preds, val_targets, val_cloud_status = [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Predicting validation set')):
            inputs = {k: v.to(device) for k, v in batch.items()
                      if k not in ['condition_weight', 'label', 'raw_label']}
            target = batch['label'].to(device)
            cloud_status = batch['cloud_status'].to(device)

            output = model(**inputs)

            val_preds.extend(output.cpu().numpy())
            val_targets.extend(target.cpu().numpy())
            val_cloud_status.extend(cloud_status.cpu().numpy())

    # 转换数据
    train_preds = np.array(train_preds).flatten()
    train_targets = np.array(train_targets).flatten()
    train_cloud_status = np.array(train_cloud_status)

    val_preds = np.array(val_preds).flatten()
    val_targets = np.array(val_targets).flatten()
    val_cloud_status = np.array(val_cloud_status)

    # 反标准化
    train_preds_denorm = train_dataset.denormalize_labels(train_preds)
    train_targets_denorm = train_dataset.denormalize_labels(train_targets)

    val_preds_denorm = val_dataset.denormalize_labels(val_preds)
    val_targets_denorm = val_dataset.denormalize_labels(val_targets)

    # 合并训练集和验证集数据
    all_preds = np.concatenate([train_preds_denorm, val_preds_denorm])
    all_targets = np.concatenate([train_targets_denorm, val_targets_denorm])
    all_cloud_status = np.concatenate([train_cloud_status, val_cloud_status])

    # 合并时间
    if train_times is not None and val_times is not None:
        all_times = np.concatenate([train_times, val_times])
    else:
        # 如果无法获取时间，使用索引作为占位符
        all_times = np.arange(len(all_preds))
        print("警告：无法获取时间信息，使用索引作为时间")

    return (all_preds, all_targets, all_cloud_status, all_times,
            train_preds_denorm, train_targets_denorm, train_cloud_status,
            val_preds_denorm, val_targets_denorm, val_cloud_status)


def calculate_metrics(preds, targets, dataset_name=""):
    """计算评估指标"""
    if len(preds) == 0 or len(targets) == 0:
        return {
            'samples': 0,
            'mse': 0, 'rmse': 0, 'mae': 0, 'bias': 0,
            'r2': 0, 'corr': 0,
            'mean_target': 0, 'std_target': 0,
            'mean_pred': 0, 'std_pred': 0,
            'std_ratio': 0, 'slope': 0,
            'mean_relative_error_percent': 0,
            'within_10_percent_percent': 0
        }

    # 计算基本指标
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, preds)
    bias = np.mean(preds - targets)
    r2 = r2_score(targets, preds)

    # 计算相关系数
    if len(preds) > 1:
        corr = np.corrcoef(preds, targets)[0, 1]
    else:
        corr = 0

    # 计算统计指标
    mean_target = np.mean(targets)
    std_target = np.std(targets)
    mean_pred = np.mean(preds)
    std_pred = np.std(preds)
    std_ratio = std_pred / (std_target + 1e-10)
    slope = np.mean((preds - mean_pred) * (targets - mean_target)) / (np.var(targets) + 1e-10)

    # 计算相对误差
    relative_error = np.abs(preds - targets) / (np.abs(targets) + 1e-10)  # 避免除以0
    mean_relative_error = np.mean(relative_error) * 100  # 转换为百分比

    # 计算在±10%误差范围内的样本比例
    within_10_percent = np.mean(relative_error <= 0.1) * 100

    return {
        'samples': len(preds),
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'r2': r2,
        'corr': corr,
        'mean_target': mean_target,
        'std_target': std_target,
        'mean_pred': mean_pred,
        'std_pred': std_pred,
        'std_ratio': std_ratio,
        'slope': slope,
        'mean_relative_error_percent': mean_relative_error,
        'within_10_percent_percent': within_10_percent
    }


def plot_train_val_comparison(train_preds, train_targets, val_preds, val_targets,
                              train_cloud_status, val_cloud_status, output_dir):
    """绘制训练集和验证集对比图"""
    print(f"\n绘制训练集和验证集对比图...")

    def calculate_plot_metrics(preds, targets):
        return {
            'rmse': np.sqrt(mean_squared_error(targets, preds)),
            'mae': mean_absolute_error(targets, preds),
            'bias': np.mean(preds - targets),
            'r2': r2_score(targets, preds)
        }

    def add_scatter_panel(ax, fig, targets, preds, cloud_status, title, metrics):
        scatter = ax.scatter(targets, preds, alpha=0.6, s=8,
                             c=cloud_status, cmap='viridis', edgecolor='none')
        fig.colorbar(scatter, ax=ax, label='Cloud Status')
        ax.plot([all_min, all_max], [all_min, all_max], 'k--', alpha=0.7)
        ax.set_xlabel('Actual PBL Height (m)')
        ax.set_ylabel('Predicted PBL Height (m)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, f"RMSE = {metrics['rmse']:.1f} m\nR² = {metrics['r2']:.3f}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def add_error_hist(ax, errors, title, label, color=None):
        hist_kwargs = {'bins': 50, 'alpha': 0.7, 'label': label, 'edgecolor': 'black'}
        if color is not None:
            hist_kwargs['color'] = color
        ax.hist(errors, **hist_kwargs)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Error (Predicted - Actual) (m)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{title}\nMean = {errors.mean():.1f} m, Std = {errors.std():.1f} m')
        ax.legend()
        ax.grid(True, alpha=0.3)

    all_min = min(train_targets.min(), train_preds.min(), val_targets.min(), val_preds.min())
    all_max = max(train_targets.max(), train_preds.max(), val_targets.max(), val_preds.max())
    train_metrics = calculate_plot_metrics(train_preds, train_targets)
    val_metrics = calculate_plot_metrics(val_preds, val_targets)

    # 1. 绘制散点图对比
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    add_scatter_panel(
        axes[0], fig, train_targets, train_preds, train_cloud_status,
        f'Training Set (n={len(train_preds)})', train_metrics
    )
    add_scatter_panel(
        axes[1], fig, val_targets, val_preds, val_cloud_status,
        f'Validation Set (n={len(val_preds)})', val_metrics
    )
    fig.tight_layout()
    scatter_path = os.path.join(output_dir, 'train_val_scatter_comparison.png')
    fig.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 2. 绘制误差分布对比
    train_errors = train_preds - train_targets
    val_errors = val_preds - val_targets

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    add_error_hist(axes[0], train_errors, 'Training Set Error Distribution', 'Training Set')
    add_error_hist(axes[1], val_errors, 'Validation Set Error Distribution', 'Validation Set', color='orange')
    fig.tight_layout()
    error_path = os.path.join(output_dir, 'train_val_error_distribution.png')
    fig.savefig(error_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3. 绘制综合对比图
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(train_targets, train_preds, alpha=0.6, s=8,
               color='blue', edgecolor='none', label=f'Training Set (n={len(train_preds)})')
    ax.scatter(val_targets, val_preds, alpha=0.6, s=8,
               color='red', edgecolor='none', label=f'Validation Set (n={len(val_preds)})')
    ax.plot([all_min, all_max], [all_min, all_max], 'k--', alpha=0.7, label='Perfect prediction')
    ax.plot([all_min, all_max], [all_min, all_max], 'k-', alpha=0.3, linewidth=0.5)

    margin = 50
    ax.set_xlim([all_min - margin, all_max + margin])
    ax.set_ylim([all_min - margin, all_max + margin])

    # 添加统计信息文本
    stats_text = f'Training Set:\n' \
                 f"RMSE = {train_metrics['rmse']:.1f} m\n" \
                 f"MAE = {train_metrics['mae']:.1f} m\n" \
                 f"Bias = {train_metrics['bias']:.1f} m\n" \
                 f"R² = {train_metrics['r2']:.3f}\n\n" \
                 f'Validation Set:\n' \
                 f"RMSE = {val_metrics['rmse']:.1f} m\n" \
                 f"MAE = {val_metrics['mae']:.1f} m\n" \
                 f"Bias = {val_metrics['bias']:.1f} m\n" \
                 f"R² = {val_metrics['r2']:.3f}"

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Actual PBL Height (m)')
    ax.set_ylabel('Predicted PBL Height (m)')
    ax.set_title('Scatter Plot: Training vs Validation Set Performance')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    combined_path = os.path.join(output_dir, 'train_val_combined_comparison.png')
    fig.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"训练集验证集对比图已保存到:")
    print(f"  {scatter_path}")
    print(f"  {error_path}")
    print(f"  {combined_path}")

    return scatter_path, error_path, combined_path


def save_train_val_results(train_preds, train_targets, train_cloud_status, train_times,
                           val_preds, val_targets, val_cloud_status, val_times,
                           train_metrics, val_metrics, combined_metrics, output_dir):
    """保存训练集和验证集的结果"""
    os.makedirs(output_dir, exist_ok=True)

    def prepare_time_data(times, preds, targets, cloud_status, dataset_name):
        """准备时间数据"""
        time_str_list = []
        for i, time_val in enumerate(times):
            if isinstance(time_val, (pd.Timestamp, np.datetime64)):
                time_str = pd.to_datetime(time_val).strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(time_val, str) and len(time_val) > 0:
                time_str = time_val
            else:
                time_str = f"{dataset_name}_{i}"
            time_str_list.append(time_str)
        return time_str_list

    # 准备训练集时间数据
    if train_times is None or len(train_times) != len(train_preds):
        train_times = [f"train_{i}" for i in range(len(train_preds))]

    train_time_str = prepare_time_data(train_times, train_preds, train_targets, train_cloud_status, "train")

    # 准备验证集时间数据
    if val_times is None or len(val_times) != len(val_preds):
        val_times = [f"val_{i}" for i in range(len(val_preds))]

    val_time_str = prepare_time_data(val_times, val_preds, val_targets, val_cloud_status, "val")

    # 创建训练集DataFrame
    train_df = pd.DataFrame({
        'time': train_time_str,
        'dataset': 'train',
        'actual_pbl_height': train_targets,
        'predicted_pbl_height': train_preds,
        'cloud_status': train_cloud_status,
        'error': train_preds - train_targets,
        'abs_error': np.abs(train_preds - train_targets),
        'relative_error_percent': np.abs(train_preds - train_targets) / (np.abs(train_targets) + 1e-10) * 100
    })

    # 创建验证集DataFrame
    val_df = pd.DataFrame({
        'time': val_time_str,
        'dataset': 'val',
        'actual_pbl_height': val_targets,
        'predicted_pbl_height': val_preds,
        'cloud_status': val_cloud_status,
        'error': val_preds - val_targets,
        'abs_error': np.abs(val_preds - val_targets),
        'relative_error_percent': np.abs(val_preds - val_targets) / (np.abs(val_targets) + 1e-10) * 100
    })

    # 合并数据集
    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    # 尝试按时间排序
    if all(isinstance(t, str) for t in combined_df['time']):
        try:
            combined_df['time_parsed'] = pd.to_datetime(combined_df['time'], errors='coerce')
            combined_df = combined_df.sort_values('time_parsed')
            combined_df = combined_df.drop('time_parsed', axis=1)
            print("结果已按时间排序")
        except:
            print("无法按时间排序，保持原始顺序")

    # 保存合并的CSV
    combined_csv_path = os.path.join(output_dir, 'train_val_predictions.csv')
    combined_df.to_csv(combined_csv_path, index=False)

    # 分别保存训练集和验证集
    train_csv_path = os.path.join(output_dir, 'train_predictions.csv')
    val_csv_path = os.path.join(output_dir, 'val_predictions.csv')

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    # 保存统计摘要
    summary_path = os.path.join(output_dir, 'train_val_summary.csv')

    summary_data = []
    for dataset_name, df in [('train', train_df), ('val', val_df), ('combined', combined_df)]:
        if len(df) > 0:
            summary_data.append({
                'dataset': dataset_name,
                'samples': len(df),
                'rmse': np.sqrt(mean_squared_error(df['actual_pbl_height'], df['predicted_pbl_height'])),
                'mae': mean_absolute_error(df['actual_pbl_height'], df['predicted_pbl_height']),
                'bias': df['error'].mean(),
                'r2': r2_score(df['actual_pbl_height'], df['predicted_pbl_height']),
                'mean_actual': df['actual_pbl_height'].mean(),
                'std_actual': df['actual_pbl_height'].std(),
                'mean_pred': df['predicted_pbl_height'].mean(),
                'std_pred': df['predicted_pbl_height'].std(),
                'mean_abs_error': df['abs_error'].mean(),
                'std_abs_error': df['abs_error'].std(),
                'mean_rel_error_percent': df['relative_error_percent'].mean()
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_path, index=False)

    # 保存指标
    metrics_path = os.path.join(output_dir, 'train_val_metrics.npz')
    np.savez(metrics_path,
             train_preds=train_preds,
             train_targets=train_targets,
             train_cloud_status=train_cloud_status,
             train_times=train_times,
             val_preds=val_preds,
             val_targets=val_targets,
             val_cloud_status=val_cloud_status,
             val_times=val_times,
             train_metrics=train_metrics,
             val_metrics=val_metrics,
             combined_metrics=combined_metrics)

    # 保存指标为文本文件
    metrics_text_path = os.path.join(output_dir, 'train_val_metrics.txt')
    with open(metrics_text_path, 'w', encoding='utf-8') as f:
        f.write("Training and Validation Set Evaluation Metrics\n")
        f.write("=" * 60 + "\n\n")

        f.write("Training Set Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Samples: {train_metrics['samples']}\n")
        f.write(f"RMSE: {train_metrics['rmse']:.2f} m\n")
        f.write(f"MAE: {train_metrics['mae']:.2f} m\n")
        f.write(f"Bias: {train_metrics['bias']:.2f} m\n")
        f.write(f"R²: {train_metrics['r2']:.3f}\n")
        f.write(f"Correlation (R): {train_metrics['corr']:.3f}\n")
        f.write(f"Mean Target: {train_metrics['mean_target']:.2f} m\n")
        f.write(f"Std Target: {train_metrics['std_target']:.2f} m\n")
        f.write(f"Mean Prediction: {train_metrics['mean_pred']:.2f} m\n")
        f.write(f"Std Prediction: {train_metrics['std_pred']:.2f} m\n")
        f.write(f"Std Ratio (Pred/Target): {train_metrics['std_ratio']:.4f}\n")
        f.write(f"Slope (Pred vs Target): {train_metrics['slope']:.4f}\n")
        f.write(f"Mean Relative Error: {train_metrics['mean_relative_error_percent']:.2f}%\n")
        f.write(f"Within ±10%: {train_metrics['within_10_percent_percent']:.2f}%\n\n")

        f.write("Validation Set Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Samples: {val_metrics['samples']}\n")
        f.write(f"RMSE: {val_metrics['rmse']:.2f} m\n")
        f.write(f"MAE: {val_metrics['mae']:.2f} m\n")
        f.write(f"Bias: {val_metrics['bias']:.2f} m\n")
        f.write(f"R²: {val_metrics['r2']:.3f}\n")
        f.write(f"Correlation (R): {val_metrics['corr']:.3f}\n")
        f.write(f"Mean Target: {val_metrics['mean_target']:.2f} m\n")
        f.write(f"Std Target: {val_metrics['std_target']:.2f} m\n")
        f.write(f"Mean Prediction: {val_metrics['mean_pred']:.2f} m\n")
        f.write(f"Std Prediction: {val_metrics['std_pred']:.2f} m\n")
        f.write(f"Std Ratio (Pred/Target): {val_metrics['std_ratio']:.4f}\n")
        f.write(f"Slope (Pred vs Target): {val_metrics['slope']:.4f}\n")
        f.write(f"Mean Relative Error: {val_metrics['mean_relative_error_percent']:.2f}%\n")
        f.write(f"Within ±10%: {val_metrics['within_10_percent_percent']:.2f}%\n\n")

        f.write("Combined Metrics (Train + Val):\n")
        f.write("-" * 40 + "\n")
        f.write(f"Samples: {combined_metrics['samples']}\n")
        f.write(f"RMSE: {combined_metrics['rmse']:.2f} m\n")
        f.write(f"MAE: {combined_metrics['mae']:.2f} m\n")
        f.write(f"Bias: {combined_metrics['bias']:.2f} m\n")
        f.write(f"R²: {combined_metrics['r2']:.3f}\n")
        f.write(f"Correlation (R): {combined_metrics['corr']:.3f}\n")
        f.write(f"Mean Target: {combined_metrics['mean_target']:.2f} m\n")
        f.write(f"Std Target: {combined_metrics['std_target']:.2f} m\n")
        f.write(f"Mean Prediction: {combined_metrics['mean_pred']:.2f} m\n")
        f.write(f"Std Prediction: {combined_metrics['std_pred']:.2f} m\n")
        f.write(f"Std Ratio (Pred/Target): {combined_metrics['std_ratio']:.4f}\n")
        f.write(f"Slope (Pred vs Target): {combined_metrics['slope']:.4f}\n")
        f.write(f"Mean Relative Error: {combined_metrics['mean_relative_error_percent']:.2f}%\n")
        f.write(f"Within ±10%: {combined_metrics['within_10_percent_percent']:.2f}%\n")

    print(f"\n结果已保存:")
    print(f"  合并数据: {combined_csv_path} ({len(combined_df)} 个样本)")
    print(f"  训练集: {train_csv_path} ({len(train_df)} 个样本)")
    print(f"  验证集: {val_csv_path} ({len(val_df)} 个样本)")
    print(f"  统计摘要: {summary_path}")
    print(f"  指标文件: {metrics_text_path}")

    return combined_csv_path, summary_path, metrics_text_path


def main_evaluate_train_val():
    """主评估函数 - 评估训练集和验证集"""
    try:
        # 设置设备
        device = get_device()
        TRAIN_CONFIG['device'] = str(device)

        # 创建输出目录
        create_output_dir(TRAIN_CONFIG)

        print("=" * 60)
        print("PBL Height Neural Network Evaluation (Training + Validation Sets)")
        print("=" * 60)
        print(f"Device: {TRAIN_CONFIG['device']}")
        print(f"Output directory: {TRAIN_CONFIG['output_dir']}")
        print(f"Features file: {TRAIN_CONFIG['features_file1']}")
        print(f"Labels file: {TRAIN_CONFIG['labels_file1']}")
        print(f"Split method: {TRAIN_CONFIG['split_method']}")
        print(f"Train ratio: {TRAIN_CONFIG['train_ratio']}")
        print(f"Validation ratio: {TRAIN_CONFIG['val_ratio']}")
        print(f"Use AERI: {TRAIN_CONFIG.get('use_aeri', True)}")
        print(f"Use microwave: {TRAIN_CONFIG.get('use_microwave', True)}")
        print(f"Use wind: {TRAIN_CONFIG.get('use_wind_features', True)}")
        print(f"Use ceilometer: {TRAIN_CONFIG.get('use_ceilometer', True)}")

        # 1. 加载训练集和验证集
        print("\n" + "=" * 60)
        print("1. Loading training and validation datasets...")
        print("=" * 60)

        training_scalers = load_training_scalers()
        datasets, data_loaders = load_train_val_datasets(training_scalers)

        if datasets is None or data_loaders is None:
            print("Failed to load datasets. Exiting.")
            return

        print(f"\n数据集加载完成:")
        print(f"  训练集: {len(datasets['train'])} 个样本")
        print(f"  验证集: {len(datasets['val'])} 个样本")
        print(f"  总计: {len(datasets['train']) + len(datasets['val'])} 个样本")

        # 2. 加载最佳模型
        print("\n" + "=" * 60)
        print("2. Loading best model...")
        print("=" * 60)

        model_path = os.path.join(TRAIN_CONFIG['output_dir'], TRAIN_CONFIG['model_save_path'])

        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            print("Please run train_main.py first to train a model.")
            return

        checkpoint = safe_load_checkpoint(model_path, weights_only=False)

        if checkpoint is None:
            print("Failed to load checkpoint")
            return

        actual_physics_dim = datasets['train'].physics_features.shape[1]
        actual_hatpro_dim = datasets['train'].hatpro.shape[1]
        actual_miracp_dim = datasets['train'].miracp.shape[1]
        actual_wind_dim = datasets['train'].u_wind.shape[1]
        actual_aeri_dim = datasets['train'].aeri_channels if TRAIN_CONFIG.get('use_aeri', True) else 0
        actual_cloud_classes = int(
            max(datasets['train'].cloud_status.max(), datasets['val'].cloud_status.max()) + 1
        )

        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Evaluation physics_dim: {actual_physics_dim}")
        print(f"Evaluation microwave dims: HATPRO={actual_hatpro_dim}, MiRAC-P={actual_miracp_dim}")
        print(f"Evaluation wind dim: {actual_wind_dim}")
        print(f"Evaluation AERI channels: {actual_aeri_dim}")
        print(f"Evaluation cloud classes: {actual_cloud_classes}")

        # 创建与训练时一致的新模型实例
        eval_model = create_pbl_model(
            aeri_dim=actual_aeri_dim,
            hatpro_dim=actual_hatpro_dim,
            miracp_dim=actual_miracp_dim,
            ceil_dim=MODEL_CONFIG['ceil_dim'],
            wind_dim=actual_wind_dim,
            use_wind_branch=TRAIN_CONFIG.get('use_wind_features', True),
            physics_dim=actual_physics_dim,
            dropout_rate=TRAIN_CONFIG['dropout_rate'],
            n_cloud_classes=actual_cloud_classes,
            n_condition_classes=MODEL_CONFIG.get('n_condition_classes', 3),
            near_surface_bins=MODEL_CONFIG.get('near_surface_bins', 48),
            condition_prior_scale=MODEL_CONFIG.get('condition_prior_scale', 1.5),
            cloudy_refine_scale=MODEL_CONFIG.get('cloudy_refine_scale', 0.22),
            hatpro_k_band_dim=MODEL_CONFIG.get('hatpro_k_band_dim', 7),
            hatpro_v_band_dim=MODEL_CONFIG.get('hatpro_v_band_dim', 7),
            miracp_absorption_dim=MODEL_CONFIG.get('miracp_absorption_dim', 6),
            miracp_window_dim=MODEL_CONFIG.get('miracp_window_dim', 2),
        )

        # 加载状态
        eval_model.load_state_dict(checkpoint['model_state_dict'])

        # 3. 对训练集和验证集进行预测
        print("\n" + "=" * 60)
        print("3. Predicting on training and validation sets...")
        print("=" * 60)

        # 获取所有预测结果
        (all_preds, all_targets, all_cloud_status, all_times,
         train_preds, train_targets, train_cloud_status,
         val_preds, val_targets, val_cloud_status) = predict_train_val_datasets(
            eval_model,
            data_loaders['train'],
            data_loaders['val'],
            datasets['train'],
            datasets['val'],
            device
        )

        # 打印基本信息
        print(f"\n数据集基本信息:")
        print(f"  训练集样本数: {len(train_preds)}")
        print(f"  验证集样本数: {len(val_preds)}")
        print(f"  总样本数: {len(all_preds)}")
        print(f"  训练集PBL高度范围: {train_targets.min():.1f} - {train_targets.max():.1f} m")
        print(f"  验证集PBL高度范围: {val_targets.min():.1f} - {val_targets.max():.1f} m")

        # 云状态分布
        print(f"\n训练集云状态分布:")
        unique_train_cloud, train_counts = np.unique(train_cloud_status, return_counts=True)
        for status, count in zip(unique_train_cloud, train_counts):
            print(f"    状态{status}: {count} 个样本 ({count / len(train_cloud_status) * 100:.1f}%)")

        print(f"验证集云状态分布:")
        unique_val_cloud, val_counts = np.unique(val_cloud_status, return_counts=True)
        for status, count in zip(unique_val_cloud, val_counts):
            print(f"    状态{status}: {count} 个样本 ({count / len(val_cloud_status) * 100:.1f}%)")

        # 4. 计算评估指标
        print("\n" + "=" * 60)
        print("4. Calculating evaluation metrics...")
        print("=" * 60)

        # 计算训练集指标
        train_metrics = calculate_metrics(train_preds, train_targets, "Training Set")
        print(f"\nTraining set performance:")
        print(f"  Samples: {train_metrics['samples']}")
        print(f"  RMSE: {train_metrics['rmse']:.2f} m")
        print(f"  MAE: {train_metrics['mae']:.2f} m")
        print(f"  Bias: {train_metrics['bias']:.2f} m")
        print(f"  R²: {train_metrics['r2']:.3f}")
        print(f"  Correlation (R): {train_metrics['corr']:.3f}")
        print(f"  Std Ratio (Pred/Target): {train_metrics['std_ratio']:.3f}")
        print(f"  Slope (Pred vs Target): {train_metrics['slope']:.3f}")
        print(f"  Mean Relative Error: {train_metrics['mean_relative_error_percent']:.2f}%")
        print(f"  Within ±10%: {train_metrics['within_10_percent_percent']:.2f}%")

        # 计算验证集指标
        val_metrics = calculate_metrics(val_preds, val_targets, "Validation Set")
        print(f"\nValidation set performance:")
        print(f"  Samples: {val_metrics['samples']}")
        print(f"  RMSE: {val_metrics['rmse']:.2f} m")
        print(f"  MAE: {val_metrics['mae']:.2f} m")
        print(f"  Bias: {val_metrics['bias']:.2f} m")
        print(f"  R²: {val_metrics['r2']:.3f}")
        print(f"  Correlation (R): {val_metrics['corr']:.3f}")
        print(f"  Std Ratio (Pred/Target): {val_metrics['std_ratio']:.3f}")
        print(f"  Slope (Pred vs Target): {val_metrics['slope']:.3f}")
        print(f"  Mean Relative Error: {val_metrics['mean_relative_error_percent']:.2f}%")
        print(f"  Within ±10%: {val_metrics['within_10_percent_percent']:.2f}%")

        # 计算合并指标
        combined_metrics = calculate_metrics(all_preds, all_targets, "Combined Set")
        print(f"\nCombined performance (Train + Val):")
        print(f"  Total samples: {combined_metrics['samples']}")
        print(f"  RMSE: {combined_metrics['rmse']:.2f} m")
        print(f"  MAE: {combined_metrics['mae']:.2f} m")
        print(f"  Bias: {combined_metrics['bias']:.2f} m")
        print(f"  R²: {combined_metrics['r2']:.3f}")
        print(f"  Correlation (R): {combined_metrics['corr']:.3f}")
        print(f"  Std Ratio (Pred/Target): {combined_metrics['std_ratio']:.3f}")
        print(f"  Slope (Pred vs Target): {combined_metrics['slope']:.3f}")
        print(f"  Mean Relative Error: {combined_metrics['mean_relative_error_percent']:.2f}%")
        print(f"  Within ±10%: {combined_metrics['within_10_percent_percent']:.2f}%")

        # 5. 绘制训练集和验证集的评估结果图
        print("\n" + "=" * 60)
        print("5. Plotting evaluation results for training and validation sets...")
        print("=" * 60)

        # 绘制训练集和验证集对比图
        plot_train_val_comparison(
            train_preds, train_targets, val_preds, val_targets,
            train_cloud_status, val_cloud_status, TRAIN_CONFIG['output_dir']
        )

        # 绘制整个数据集的评估图表
        plot_evaluation_results(all_preds, all_targets, all_cloud_status, TRAIN_CONFIG['output_dir'])

        # 增强散点图
        plot_enhanced_scatter_plots(all_preds, all_targets, all_cloud_status, TRAIN_CONFIG['output_dir'])

        # 6. 绘制训练历史（如果存在）
        print("\n" + "=" * 60)
        print("6. Plotting training history...")
        print("=" * 60)

        history_path = os.path.join(TRAIN_CONFIG['output_dir'], 'training_history.png')
        if not os.path.exists(history_path) and 'history' in checkpoint:
            if 'history' in checkpoint:
                plot_training_history(checkpoint['history'], TRAIN_CONFIG['output_dir'])
                print(f"Training history plot saved to: {history_path}")
            else:
                print("No training history found in checkpoint")
        else:
            print(f"Training history already exists or not available: {history_path}")

        # 7. 保存训练集和验证集的结果
        print("\n" + "=" * 60)
        print("7. Saving training and validation set results...")
        print("=" * 60)

        # 获取时间信息
        train_times = datasets['train'].get_all_times() if hasattr(datasets['train'], 'get_all_times') else None
        val_times = datasets['val'].get_all_times() if hasattr(datasets['val'], 'get_all_times') else None

        # 保存结果
        combined_csv_path, summary_path, metrics_text_path = save_train_val_results(
            train_preds, train_targets, train_cloud_status, train_times,
            val_preds, val_targets, val_cloud_status, val_times,
            train_metrics, val_metrics, combined_metrics, TRAIN_CONFIG['output_dir']
        )

        # 8. 保存最终报告
        print("\n" + "=" * 60)
        print("8. Generating evaluation report...")
        print("=" * 60)

        report_path = os.path.join(TRAIN_CONFIG['output_dir'], 'evaluation_report_train_val.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("PBL Height Model Evaluation Report (Training + Validation Sets)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Evaluation date: {pd.Timestamp.now()}\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Dataset: {TRAIN_CONFIG['features_file1']}\n")
            f.write(f"Training samples: {len(train_preds)}\n")
            f.write(f"Validation samples: {len(val_preds)}\n")
            f.write(f"Total samples: {len(all_preds)}\n")
            f.write(f"Split method: {TRAIN_CONFIG['split_method']}\n")
            f.write(f"Train ratio: {TRAIN_CONFIG['train_ratio']}\n")
            f.write(f"Validation ratio: {TRAIN_CONFIG['val_ratio']}\n\n")

            f.write("Training Set Metrics:\n")
            f.write("-" * 40 + "\n")
            for key, value in train_metrics.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")

            f.write("\nValidation Set Metrics:\n")
            f.write("-" * 40 + "\n")
            for key, value in val_metrics.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")

            f.write("\nCombined Metrics (Train + Val):\n")
            f.write("-" * 40 + "\n")
            for key, value in combined_metrics.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("Generated Files:\n")
            f.write("-" * 40 + "\n")
            f.write(f"1. {os.path.basename(combined_csv_path)} - Combined prediction data (train + val)\n")
            f.write(f"2. train_predictions.csv - Training set predictions\n")
            f.write(f"3. val_predictions.csv - Validation set predictions\n")
            f.write(f"4. {os.path.basename(summary_path)} - Statistical summary\n")
            f.write(f"5. {os.path.basename(metrics_text_path)} - Detailed metrics\n")
            f.write("6. train_val_scatter_comparison.png - Scatter plot comparison\n")
            f.write("7. train_val_error_distribution.png - Error distribution comparison\n")
            f.write("8. train_val_combined_comparison.png - Combined scatter plot\n")
            f.write("9. evaluation_plots.png - Comprehensive evaluation plots\n")
            f.write("10. scatter_all_conditions.png - Combined scatter plot\n")
            f.write("11. scatter_by_condition.png - Individual scatter plots by condition\n")
            f.write("12. scatter_density.png - Density scatter plots\n")
            f.write("13. error_distribution.png - Error distribution by condition\n")
            f.write("14. training_history.png - Training history (if available)\n")

        print(f"\nEvaluation report saved to: {report_path}")
        print(f"\nAll evaluation results saved to: {TRAIN_CONFIG['output_dir']}")
        print("\nGenerated files:")
        print("  train_val_predictions.csv - 合并的预测数据（训练集+验证集）")
        print("  train_predictions.csv - 训练集预测数据")
        print("  val_predictions.csv - 验证集预测数据")
        print("  train_val_summary.csv - 统计摘要")
        print("  train_val_metrics.txt - 详细评估指标")
        print("  train_val_scatter_comparison.png - 训练集验证集散点图对比")
        print("  train_val_error_distribution.png - 训练集验证集误差分布对比")
        print("  train_val_combined_comparison.png - 训练集验证集综合对比图")

    except Exception as e:
        print(f"Error in evaluation: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main_evaluate_train_val()
