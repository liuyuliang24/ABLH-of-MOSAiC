# evaluate_val_only.py
"""
仅评估验证集 - 不包含训练集
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

matplotlib.use('Agg')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TRAIN_CONFIG, MODEL_CONFIG
from utils import create_output_dir, get_device, safe_load_checkpoint
from dataset import PBLDatasetFixed
from models import create_pbl_model
from plots import plot_evaluation_results, plot_enhanced_scatter_plots


def load_training_scalers():
    """加载训练阶段保存的 scaler（评估必须复用）"""
    scaler_path = os.path.join(TRAIN_CONFIG['output_dir'], TRAIN_CONFIG['scaler_save_path'])
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Training scaler file not found: {scaler_path}. "
            f"Please run train_main.py successfully before evaluation."
        )
    print(f"Loading training scalers from: {scaler_path}")
    return joblib.load(scaler_path)


def load_val_dataset(external_scalers):
    """仅加载验证集"""
    print("\nLoading validation dataset...")
    use_aeri = TRAIN_CONFIG.get('use_aeri', True)
    use_microwave = TRAIN_CONFIG.get('use_microwave', True)
    use_wind_features = TRAIN_CONFIG.get('use_wind_features', True)
    use_ceilometer = TRAIN_CONFIG.get('use_ceilometer', True)
    aeri_channels = TRAIN_CONFIG.get('aeri_channels', 0) if use_aeri else 0
    try:
        val_dataset = PBLDatasetFixed(
            features_file=TRAIN_CONFIG['features_file'],
            labels_file=TRAIN_CONFIG['labels_file'],
            mode='val',  # 仅验证集
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
        print(f"Validation set size: {len(val_dataset)} samples")
        return val_dataset
    except Exception as e:
        print(f"Error loading validation dataset: {e}")
        traceback.print_exc()
        return None


def predict_val_dataset(model, val_loader, val_dataset, device):
    """对验证集进行预测，返回物理单位的结果"""
    model = model.to(device)
    model.eval()

    val_preds, val_targets, val_cloud_status = [], [], []
    val_times = val_dataset.get_all_times() if hasattr(val_dataset, 'get_all_times') else None

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Predicting validation set'):
            inputs = {k: v.to(device) for k, v in batch.items()
                      if k not in ['condition_weight', 'label', 'raw_label']}
            target = batch['label'].to(device)
            cloud_status = batch['cloud_status'].to(device)

            output = model(**inputs)

            val_preds.extend(output.cpu().numpy())
            val_targets.extend(target.cpu().numpy())
            val_cloud_status.extend(cloud_status.cpu().numpy())

    val_preds = np.array(val_preds).flatten()
    val_targets = np.array(val_targets).flatten()
    val_cloud_status = np.array(val_cloud_status)

    # 反标准化到物理单位（米）
    val_preds_phys = val_dataset.denormalize_labels(val_preds)
    val_targets_phys = val_dataset.denormalize_labels(val_targets)

    return val_preds_phys, val_targets_phys, val_cloud_status, val_times


def calculate_metrics(preds, targets, dataset_name=""):
    """计算评估指标"""
    if len(preds) == 0:
        return {'samples': 0}

    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, preds)
    bias = np.mean(preds - targets)
    r2 = r2_score(targets, preds)
    corr = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0

    mean_target = np.mean(targets)
    std_target = np.std(targets)
    mean_pred = np.mean(preds)
    std_pred = np.std(preds)
    std_ratio = std_pred / (std_target + 1e-10)
    slope = np.mean((preds - mean_pred) * (targets - mean_target)) / (np.var(targets) + 1e-10)

    relative_error = np.abs(preds - targets) / (np.abs(targets) + 1e-10)
    mean_relative_error_percent = np.mean(relative_error) * 100
    within_10_percent_percent = np.mean(relative_error <= 0.1) * 100

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
        'mean_relative_error_percent': mean_relative_error_percent,
        'within_10_percent_percent': within_10_percent_percent
    }


def save_val_results(preds, targets, cloud_status, times, metrics, output_dir):
    """保存验证集预测结果和统计信息"""
    os.makedirs(output_dir, exist_ok=True)

    # 准备时间字符串
    time_str_list = []
    if times is not None and len(times) == len(preds):
        for t in times:
            if isinstance(t, (pd.Timestamp, np.datetime64)):
                time_str_list.append(pd.to_datetime(t).strftime('%Y-%m-%d %H:%M:%S'))
            elif isinstance(t, str):
                time_str_list.append(t)
            else:
                time_str_list.append(str(t))
    else:
        time_str_list = [f"val_{i}" for i in range(len(preds))]

    # 创建DataFrame
    df = pd.DataFrame({
        'time': time_str_list,
        'actual_pbl_height': targets,
        'predicted_pbl_height': preds,
        'cloud_status': cloud_status,
        'error': preds - targets,
        'abs_error': np.abs(preds - targets),
        'relative_error_percent': np.abs(preds - targets) / (np.abs(targets) + 1e-10) * 100
    })

    # 按时间排序（如果可以解析）
    try:
        df['time_parsed'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.sort_values('time_parsed')
        df = df.drop('time_parsed', axis=1)
        print("结果已按时间排序")
    except:
        print("无法按时间排序，保持原始顺序")

    # 保存CSV
    csv_path = os.path.join(output_dir, 'val_predictions.csv')
    df.to_csv(csv_path, index=False)
    print(f"验证集预测数据已保存: {csv_path}")

    # 保存统计摘要
    summary_path = os.path.join(output_dir, 'val_summary.csv')
    summary_df = pd.DataFrame([{
        'dataset': 'validation',
        'samples': metrics['samples'],
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'bias': metrics['bias'],
        'r2': metrics['r2'],
        'corr': metrics['corr'],
        'mean_actual': metrics['mean_target'],
        'std_actual': metrics['std_target'],
        'mean_pred': metrics['mean_pred'],
        'std_pred': metrics['std_pred'],
        'std_ratio': metrics['std_ratio'],
        'slope': metrics['slope'],
        'mean_rel_error_percent': metrics['mean_relative_error_percent'],
        'within_10_percent_percent': metrics['within_10_percent_percent']
    }])
    summary_df.to_csv(summary_path, index=False)
    print(f"统计摘要已保存: {summary_path}")

    # 保存指标文本
    metrics_path = os.path.join(output_dir, 'val_metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("Validation Set Evaluation Metrics\n")
        f.write("=" * 60 + "\n\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
    print(f"指标文本已保存: {metrics_path}")

    # 保存npz
    npz_path = os.path.join(output_dir, 'val_metrics.npz')
    np.savez(npz_path,
             preds=preds,
             targets=targets,
             cloud_status=cloud_status,
             times=times,
             metrics=metrics)
    print(f"npz文件已保存: {npz_path}")

    return csv_path, summary_path, metrics_path


def plot_val_only(preds, targets, cloud_status, output_dir):
    """生成仅验证集的评估图表"""
    print("\n绘制验证集评估图表...")

    # 复用原项目的整体评估图（传入验证集数据）
    plot_evaluation_results(preds, targets, cloud_status, output_dir)

    # 复用增强散点图
    plot_enhanced_scatter_plots(preds, targets, cloud_status, output_dir)

    # 额外绘制单独的验证集散点图和误差分布（可选）
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 散点图
    ax1 = axes[0]
    scatter = ax1.scatter(targets, preds, c=cloud_status, cmap='viridis',
                          alpha=0.6, s=10, edgecolor='none')
    fig.colorbar(scatter, ax=ax1, label='Cloud Status')
    ax1.plot([targets.min(), targets.max()], [targets.min(), targets.max()],
             'k--', alpha=0.7, label='Perfect prediction')
    ax1.set_xlabel('Actual PBL Height (m)')
    ax1.set_ylabel('Predicted PBL Height (m)')
    ax1.set_title(f'Validation Set (n={len(preds)})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 误差分布直方图
    ax2 = axes[1]
    errors = preds - targets
    ax2.hist(errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Error (Predicted - Actual) (m)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Error Distribution\nMean = {errors.mean():.1f} m, Std = {errors.std():.1f} m')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Validation Set Evaluation', fontsize=14)
    plt.tight_layout()
    val_plot_path = os.path.join(output_dir, 'validation_scatter_error.png')
    plt.savefig(val_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"验证集散点图和误差分布已保存: {val_plot_path}")


def main_evaluate_val_only():
    """主函数 - 仅评估验证集"""
    try:
        device = get_device()
        TRAIN_CONFIG['device'] = str(device)
        create_output_dir(TRAIN_CONFIG)

        print("=" * 60)
        print("PBL Height Model Evaluation - Validation Set Only")
        print("=" * 60)
        print(f"Device: {TRAIN_CONFIG['device']}")
        print(f"Output directory: {TRAIN_CONFIG['output_dir']}")
        print(f"Features file: {TRAIN_CONFIG['features_file1']}")
        print(f"Labels file: {TRAIN_CONFIG['labels_file1']}")
        print(f"Use AERI: {TRAIN_CONFIG.get('use_aeri', True)}")
        print(f"Use microwave: {TRAIN_CONFIG.get('use_microwave', True)}")
        print(f"Use wind: {TRAIN_CONFIG.get('use_wind_features', True)}")
        print(f"Use ceilometer: {TRAIN_CONFIG.get('use_ceilometer', True)}")

        # 1. 加载标准化器
        print("\n" + "=" * 60)
        print("1. Loading training scalers...")
        print("=" * 60)
        training_scalers = load_training_scalers()

        # 2. 加载验证集
        print("\n" + "=" * 60)
        print("2. Loading validation dataset...")
        print("=" * 60)
        val_dataset = load_val_dataset(training_scalers)
        if val_dataset is None:
            return

        val_loader = DataLoader(
            val_dataset,
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=False,
            num_workers=0
        )

        # 3. 加载模型
        print("\n" + "=" * 60)
        print("3. Loading best model...")
        print("=" * 60)
        model_path = os.path.join(TRAIN_CONFIG['output_dir'], TRAIN_CONFIG['model_save_path'])
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return

        checkpoint = safe_load_checkpoint(model_path, weights_only=False)
        if checkpoint is None:
            return

        actual_physics_dim = val_dataset.physics_features.shape[1]
        actual_hatpro_dim = val_dataset.hatpro.shape[1]
        actual_miracp_dim = val_dataset.miracp.shape[1]
        actual_wind_dim = val_dataset.u_wind.shape[1]
        actual_aeri_dim = val_dataset.aeri_channels if TRAIN_CONFIG.get('use_aeri', True) else 0
        actual_cloud_classes = int(val_dataset.cloud_status.max() + 1)

        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(
            f"Physics dim: {actual_physics_dim}, "
            f"HATPRO dim: {actual_hatpro_dim}, MiRAC-P dim: {actual_miracp_dim}, "
            f"Wind dim: {actual_wind_dim}, AERI channels: {actual_aeri_dim}, "
            f"Cloud classes: {actual_cloud_classes}"
        )

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
        eval_model.load_state_dict(checkpoint['model_state_dict'])

        # 4. 预测验证集
        print("\n" + "=" * 60)
        print("4. Predicting on validation set...")
        print("=" * 60)
        preds, targets, cloud_status, times = predict_val_dataset(
            eval_model, val_loader, val_dataset, device
        )

        print(f"\nValidation set size: {len(preds)}")
        print(f"PBL height range: {targets.min():.1f} - {targets.max():.1f} m")

        # 云状态分布
        print("\nCloud status distribution:")
        unique_cloud, counts = np.unique(cloud_status, return_counts=True)
        for status, cnt in zip(unique_cloud, counts):
            print(f"  Status {status}: {cnt} samples ({cnt/len(cloud_status)*100:.1f}%)")

        # 5. 计算指标
        print("\n" + "=" * 60)
        print("5. Calculating metrics...")
        print("=" * 60)
        metrics = calculate_metrics(preds, targets, "Validation Set")
        print(f"\nValidation set performance:")
        print(f"  Samples: {metrics['samples']}")
        print(f"  RMSE: {metrics['rmse']:.2f} m")
        print(f"  MAE: {metrics['mae']:.2f} m")
        print(f"  Bias: {metrics['bias']:.2f} m")
        print(f"  R²: {metrics['r2']:.3f}")
        print(f"  Correlation (R): {metrics['corr']:.3f}")
        print(f"  Std Ratio (Pred/Target): {metrics['std_ratio']:.3f}")
        print(f"  Slope (Pred vs Target): {metrics['slope']:.3f}")
        print(f"  Mean Relative Error: {metrics['mean_relative_error_percent']:.2f}%")
        print(f"  Within ±10%: {metrics['within_10_percent_percent']:.2f}%")

        # 6. 保存结果
        print("\n" + "=" * 60)
        print("6. Saving results...")
        print("=" * 60)
        save_val_results(preds, targets, cloud_status, times, metrics, TRAIN_CONFIG['output_dir'])

        # 7. 绘制验证集图表
        print("\n" + "=" * 60)
        print("7. Plotting validation set figures...")
        print("=" * 60)
        plot_val_only(preds, targets, cloud_status, TRAIN_CONFIG['output_dir'])

        # 8. 生成报告
        print("\n" + "=" * 60)
        print("8. Generating evaluation report...")
        print("=" * 60)
        report_path = os.path.join(TRAIN_CONFIG['output_dir'], 'validation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("PBL Height Model Evaluation Report (Validation Set Only)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Evaluation date: {pd.Timestamp.now()}\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Dataset: {TRAIN_CONFIG['features_file1']}\n")
            f.write(f"Validation samples: {len(preds)}\n")
            f.write(f"Split method: {TRAIN_CONFIG['split_method']}\n")
            f.write(f"Validation ratio: {TRAIN_CONFIG['val_ratio']}\n\n")

            f.write("Validation Set Metrics:\n")
            f.write("-" * 40 + "\n")
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("Generated Files:\n")
            f.write("-" * 40 + "\n")
            f.write("1. val_predictions.csv - Validation set predictions\n")
            f.write("2. val_summary.csv - Statistical summary\n")
            f.write("3. val_metrics.txt - Detailed metrics\n")
            f.write("4. val_metrics.npz - Metrics archive\n")
            f.write("5. validation_scatter_error.png - Scatter and error distribution\n")
            f.write("6. evaluation_plots.png - Comprehensive evaluation plots\n")
            f.write("7. scatter_all_conditions.png - Combined scatter plot\n")
            f.write("8. scatter_by_condition.png - Scatter plots by condition\n")
            f.write("9. scatter_density.png - Density scatter plots\n")
            f.write("10. error_distribution.png - Error distribution by condition\n")

        print(f"\nEvaluation report saved to: {report_path}")
        print(f"\nAll validation results saved to: {TRAIN_CONFIG['output_dir']}")

    except Exception as e:
        print(f"Error in validation evaluation: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main_evaluate_val_only()
