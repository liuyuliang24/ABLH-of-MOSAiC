# evaluate.py
"""
评估函数模块
"""
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import torch


def evaluate_model(model, data_loader, dataset, config, save_results=True):
    """Evaluate model with enhanced scatter plots"""
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(device)
    model.eval()

    all_preds, all_targets, all_cloud_status = [], [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating model'):
            # 提取输入特征
            hatpro = batch['hatpro'].to(device)
            miracp = batch['miracp'].to(device)
            backscatter = batch['backscatter'].to(device)
            u_wind = batch['u_wind'].to(device)
            v_wind = batch['v_wind'].to(device)
            physics = batch['physics'].to(device)
            cloud_status = batch['cloud_status'].to(device)
            condition_label = batch['condition_label'].to(device) if 'condition_label' in batch else None
            target = batch['label'].to(device)

            # 检查是否有AERI数据
            aeri_rad = None
            if 'aeri_rad' in batch and batch['aeri_rad'].numel() > 0:
                aeri_rad = batch['aeri_rad'].to(device)

            # 准备模型输入
            model_inputs = {
                'hatpro': hatpro,
                'miracp': miracp,
                'backscatter': backscatter,
                'u_wind': u_wind,
                'v_wind': v_wind,
                'physics': physics,
                'cloud_status': cloud_status
            }

            if condition_label is not None:
                model_inputs['condition_label'] = condition_label

            # 只在有AERI数据时添加
            if aeri_rad is not None:
                model_inputs['aeri_rad'] = aeri_rad

            output = model(**model_inputs)

            all_preds.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_cloud_status.extend(cloud_status.cpu().numpy())

    # Convert data
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    all_cloud_status = np.array(all_cloud_status)

    # Denormalize
    preds_denorm = dataset.denormalize_labels(all_preds)
    targets_denorm = dataset.denormalize_labels(all_targets)

    # Calculate metrics
    metrics = calculate_metrics(preds_denorm, targets_denorm, all_cloud_status)

    # Print results
    print_evaluation_results(metrics)

    # Save and plot
    if save_results:
        output_dir = config.get('output_dir', './results')
        os.makedirs(output_dir, exist_ok=True)

        # Save results
        save_evaluation_results(preds_denorm, targets_denorm, all_cloud_status, metrics, output_dir)

    return metrics, preds_denorm, targets_denorm, all_cloud_status


def calculate_metrics(preds, targets, cloud_status):
    """Calculate evaluation metrics"""
    # Overall metrics
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, preds)
    bias = np.mean(preds - targets)
    r2 = r2_score(targets, preds)
    corr = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0
    std_target = np.std(targets)
    std_pred = np.std(preds)
    std_ratio = std_pred / (std_target + 1e-10)
    slope = np.mean((preds - preds.mean()) * (targets - targets.mean())) / (np.var(targets) + 1e-10)

    # Condition metrics
    condition_metrics = {}
    condition_names = ['Clear', 'Cloudy', 'Fog/Mist']
    condition_masks = [
        cloud_status == 0,
        (cloud_status >= 1) & (cloud_status <= 3),
        cloud_status >= 4
    ]

    for name, mask in zip(condition_names, condition_masks):
        if np.sum(mask) > 0:
            cond_pred = preds[mask]
            cond_target = targets[mask]

            cond_metrics = {
                'count': int(np.sum(mask)),
                'rmse': np.sqrt(mean_squared_error(cond_target, cond_pred)),
                'mae': mean_absolute_error(cond_target, cond_pred),
                'bias': np.mean(cond_pred - cond_target),
                'r2': r2_score(cond_target, cond_pred) if len(cond_pred) > 1 else 0
            }
            condition_metrics[name] = cond_metrics

    return {
        'overall': {
            'mse': mse, 'rmse': rmse, 'mae': mae,
            'bias': bias, 'r2': r2, 'corr': corr,
            'std_ratio': std_ratio, 'slope': slope
        },
        'conditions': condition_metrics,
        'predictions': preds,
        'targets': targets,
        'cloud_status': cloud_status
    }


def print_evaluation_results(metrics):
    """Print evaluation results"""
    overall = metrics['overall']
    conditions = metrics['conditions']

    print("=" * 60)
    print("Model Evaluation Results")
    print("=" * 60)
    print(f"Number of samples: {len(metrics['predictions'])}")
    print(f"MSE: {overall['mse']:.2f} m²")
    print(f"RMSE: {overall['rmse']:.2f} m")
    print(f"MAE: {overall['mae']:.2f} m")
    print(f"Bias: {overall['bias']:.2f} m")
    print(f"Correlation (R): {overall['corr']:.3f}")
    print(f"R² score: {overall['r2']:.3f}")
    print(f"Std Ratio (Pred/Target): {overall['std_ratio']:.3f}")
    print(f"Slope (Pred vs Target): {overall['slope']:.3f}")
    print(f"Prediction range: [{metrics['predictions'].min():.2f}, {metrics['predictions'].max():.2f}] m")
    print(f"Target range: [{metrics['targets'].min():.2f}, {metrics['targets'].max():.2f}] m")

    print("\nEvaluation by weather condition:")
    print("-" * 40)
    for name, cond_metrics in conditions.items():
        print(f"\n{name}: {cond_metrics['count']} samples")
        print(f"  RMSE: {cond_metrics['rmse']:.2f} m")
        print(f"  MAE: {cond_metrics['mae']:.2f} m")
        print(f"  Bias: {cond_metrics['bias']:.2f} m")
        print(f"  R²: {cond_metrics['r2']:.3f}")


def save_evaluation_results(preds, targets, cloud_status, metrics, output_dir):
    """Save evaluation results"""
    # Save as CSV
    results_df = pd.DataFrame({
        'prediction': preds,
        'target': targets,
        'cloud_status': cloud_status,
        'error': preds - targets,
        'abs_error': np.abs(preds - targets)
    })

    csv_path = os.path.join(output_dir, 'predictions.csv')
    results_df.to_csv(csv_path, index=False)

    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.npz')
    np.savez(metrics_path,
             predictions=preds,
             targets=targets,
             cloud_status=cloud_status,
             **metrics['overall'])

    print(f"\nResults saved to: {output_dir}")
