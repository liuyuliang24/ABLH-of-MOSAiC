# scatter_plots.py
"""
专门的散点图模块
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from plot_font_utils import get_available_serif_font


def _set_density_style() -> None:
    serif_font = get_available_serif_font()
    plt.rcParams.update(
        {
            "font.family": serif_font,
            "font.size": 10.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
        }
    )


def create_scatter_plots(preds, targets, cloud_status, output_dir, prefix="test"):
    """
    创建完整的散点图系列

    参数:
        preds: 预测值
        targets: 真实值
        cloud_status: 云状态
        output_dir: 输出目录
        prefix: 文件名前缀
    """

    # 创建条件掩码
    clear_mask = cloud_status == 0
    cloudy_mask = (cloud_status >= 1) & (cloud_status <= 3)
    fog_mask = cloud_status >= 4

    conditions = [
        ("All", np.ones_like(cloud_status, dtype=bool), "gray"),
        ("Clear", clear_mask, "blue"),
        ("Cloudy", cloudy_mask, "green"),
        ("Fog/Mist", fog_mask, "red")
    ]

    # 1. 综合散点图
    plot_combined_scatter(preds, targets, cloud_status, conditions, output_dir, prefix)

    # 2. 分条件散点图
    plot_individual_scatter(preds, targets, conditions, output_dir, prefix)

    # 3. 密度散点图
    plot_density_scatter(preds, targets, conditions, output_dir, prefix)

    # 4. 误差分布图
    plot_error_distribution(preds, targets, cloud_status, output_dir, prefix)


def plot_combined_scatter(preds, targets, cloud_status, conditions, output_dir, prefix):
    """绘制综合散点图"""
    plt.figure(figsize=(10, 8))

    for name, mask, color in conditions:
        if np.sum(mask) > 0:
            cond_targets = targets[mask]
            cond_preds = preds[mask]
            plt.scatter(cond_targets, cond_preds, alpha=0.6, s=10,
                        color=color, label=f'{name} (n={len(cond_targets)})')

    # 添加完美预测线
    all_min = min(targets.min(), preds.min())
    all_max = max(targets.max(), preds.max())
    plt.plot([all_min, all_max], [all_min, all_max], 'k--', alpha=0.7, label='Perfect')
    plt.plot([all_min, all_max], [all_min, all_max], 'k-', alpha=0.3, linewidth=0.5)

    plt.xlabel('Actual PBL Height (m)')
    plt.ylabel('Predicted PBL Height (m)')
    plt.title(f'{prefix} - Predicted vs Actual (All Conditions)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'{prefix}_scatter_combined.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Combined scatter plot saved: {save_path}")


def plot_individual_scatter(preds, targets, conditions, output_dir, prefix):
    """绘制分条件散点图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (name, mask, color) in enumerate(conditions):
        if np.sum(mask) > 0:
            ax = axes[idx // 2, idx % 2]
            cond_targets = targets[mask]
            cond_preds = preds[mask]

            ax.scatter(cond_targets, cond_preds, alpha=0.6, s=10, color=color)

            # 计算统计指标
            rmse = np.sqrt(mean_squared_error(cond_targets, cond_preds))
            mae = mean_absolute_error(cond_targets, cond_preds)
            bias = np.mean(cond_preds - cond_targets)
            r2 = r2_score(cond_targets, cond_preds) if len(cond_preds) > 1 else 0

            # 添加参考线
            cond_min = min(cond_targets.min(), cond_preds.min())
            cond_max = max(cond_targets.max(), cond_preds.max())
            ax.plot([cond_min, cond_max], [cond_min, cond_max], 'k--', alpha=0.7)
            ax.plot([cond_min, cond_max], [cond_min, cond_max], 'k-', alpha=0.3, linewidth=0.5)

            # 添加统计信息
            stats_text = f'n={len(cond_targets)}\nRMSE={rmse:.1f}m\nMAE={mae:.1f}m\nBias={bias:.1f}m\nR²={r2:.3f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel('Actual (m)')
            ax.set_ylabel('Predicted (m)')
            ax.set_title(f'{name} Condition')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{prefix}_scatter_individual.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Individual scatter plots saved: {save_path}")


def plot_density_scatter(preds, targets, conditions, output_dir, prefix):
    """绘制密度散点图"""
    _set_density_style()
    fig_width_in = 14.0 / 2.54
    fig_height_in = 11.7
    fig, axes = plt.subplots(2, 2, figsize=(fig_width_in, fig_height_in))

    for idx, (name, mask, color) in enumerate(conditions):
        if np.sum(mask) > 0:
            ax = axes[idx // 2, idx % 2]
            cond_targets = targets[mask]
            cond_preds = preds[mask]

            # 创建2D直方图
            hb = ax.hexbin(cond_targets, cond_preds, gridsize=30, cmap='viridis',
                           bins='log', mincnt=1)

            # 添加颜色条
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label('log10(count)', fontsize=10.5)
            cb.ax.tick_params(labelsize=10.5)

            # 添加参考线
            cond_min = min(cond_targets.min(), cond_preds.min())
            cond_max = max(cond_targets.max(), cond_preds.max())
            ax.plot([cond_min, cond_max], [cond_min, cond_max], 'r--', alpha=0.7, linewidth=1.5)
            ax.plot([cond_min, cond_max], [cond_min, cond_max], 'k-', alpha=0.3, linewidth=0.5)

            # 添加统计信息
            rmse = np.sqrt(mean_squared_error(cond_targets, cond_preds))
            r2 = r2_score(cond_targets, cond_preds) if len(cond_preds) > 1 else 0
            stats_text = f'n={len(cond_targets)}\nRMSE={rmse:.1f}m\nR²={r2:.3f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=10.5,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlabel('Actual (m)')
            ax.set_ylabel('Predicted (m)')
            ax.set_title(f'{name} (Density)')
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{prefix}_scatter_density.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Density scatter plots saved: {save_path}")


def plot_error_distribution(preds, targets, cloud_status, output_dir, prefix):
    """绘制误差分布图"""
    errors = preds - targets

    # 按条件分组误差
    clear_errors = errors[cloud_status == 0]
    cloudy_errors = errors[(cloud_status >= 1) & (cloud_status <= 3)]
    fog_errors = errors[cloud_status >= 4]

    error_data = [errors, clear_errors, cloudy_errors, fog_errors]
    labels = ['All', 'Clear', 'Cloudy', 'Fog/Mist']
    colors = ['gray', 'blue', 'green', 'red']

    plt.figure(figsize=(10, 6))

    # 创建箱线图
    bp = plt.boxplot(error_data, labels=labels, patch_artist=True)

    # 设置颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # 添加统计信息
    for i, err_data in enumerate(error_data):
        if len(err_data) > 0:
            mean_err = np.mean(err_data)
            std_err = np.std(err_data)
            median_err = np.median(err_data)

            stats_text = f'Mean: {mean_err:.1f}m\nStd: {std_err:.1f}m\nMed: {median_err:.1f}m'
            plt.text(i + 0.8, plt.ylim()[1] * 0.9, stats_text, fontsize=8,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    plt.ylabel('Prediction Error (m)')
    plt.title(f'{prefix} - Error Distribution by Condition')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{prefix}_error_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Error distribution plot saved: {save_path}")
