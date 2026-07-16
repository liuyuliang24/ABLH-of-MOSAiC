# plots.py
"""
可视化函数模块
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def _set_density_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
        }
    )


def plot_enhanced_scatter_plots(preds, targets, cloud_status, output_dir):
    """Plot enhanced scatter plots including all conditions and individual conditions"""
    _set_density_style()

    # Create condition masks
    clear_mask = cloud_status == 0
    cloudy_mask = (cloud_status >= 1) & (cloud_status <= 3)
    fog_mask = cloud_status >= 4

    # Prepare data for each condition
    conditions_data = {
        'All': (targets, preds, cloud_status, 'k', 'All'),
        'Clear': (targets[clear_mask], preds[clear_mask],
                  cloud_status[clear_mask], 'blue', 'Clear'),
        'Cloudy': (targets[cloudy_mask], preds[cloudy_mask],
                   cloud_status[cloudy_mask], 'green', 'Cloudy'),
        'Fog/Mist': (targets[fog_mask], preds[fog_mask],
                     cloud_status[fog_mask], 'red', 'Fog/Mist')
    }

    # Plot 1: Combined scatter plot for all conditions
    plt.figure(figsize=(12, 10))

    # Plot all conditions with different colors
    for condition_name, (cond_targets, cond_preds, _, color, label) in conditions_data.items():
        if len(cond_targets) > 0:
            plt.scatter(cond_targets, cond_preds, alpha=0.6, s=10,
                        color=color, label=f'{label} (n={len(cond_targets)})')

    # Add perfect prediction line
    all_min = min(targets.min(), preds.min())
    all_max = max(targets.max(), preds.max())
    plt.plot([all_min, all_max], [all_min, all_max], 'k--', alpha=0.7, label='Perfect prediction')

    # Add 1:1 line
    plt.plot([all_min, all_max], [all_min, all_max], 'k-', alpha=0.3, linewidth=0.5)

    plt.xlabel('Actual PBL Height (m)')
    plt.ylabel('Predicted PBL Height (m)')
    plt.title('Scatter Plot: Predicted vs Actual PBL Height (All Conditions)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save combined scatter plot
    combined_path = os.path.join(output_dir, 'scatter_all_conditions.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined scatter plot saved to: {combined_path}")

    # Plot 2: Individual scatter plots for each condition
    fig_width_in = 14.0 / 2.54
    fig_height_in = 14.0 / 2.54
    fig, axes = plt.subplots(2, 2, figsize=(fig_width_in, fig_height_in))

    colors = ['blue', 'green', 'red', 'purple']

    for idx, (condition_name, (cond_targets, cond_preds, _, color, label)) in enumerate(conditions_data.items()):
        if len(cond_targets) > 0:
            ax = axes[idx // 2, idx % 2]

            # Scatter plot
            scatter = ax.scatter(cond_targets, cond_preds, alpha=0.6, s=10,
                                 color=color, edgecolor='none')

            # Calculate statistics
            rmse = np.sqrt(mean_squared_error(cond_targets, cond_preds))
            mae = mean_absolute_error(cond_targets, cond_preds)
            bias = np.mean(cond_preds - cond_targets)
            r2 = r2_score(cond_targets, cond_preds) if len(cond_preds) > 1 else 0

            # Add perfect prediction line
            cond_min = min(cond_targets.min(), cond_preds.min())
            cond_max = max(cond_targets.max(), cond_preds.max())
            ax.plot([cond_min, cond_max], [cond_min, cond_max], 'k--', alpha=0.7)

            # Add 1:1 line
            ax.plot([cond_min, cond_max], [cond_min, cond_max], 'k-', alpha=0.3, linewidth=0.5)

            # Set limits
            ax.set_xlim([cond_min - 50, cond_max + 50])
            ax.set_ylim([cond_min - 50, cond_max + 50])

            # Add statistics text
            stats_text = f'n = {len(cond_targets)}\nRMSE = {rmse:.1f} m\nMAE = {mae:.1f} m\nBias = {bias:.1f} m\nR² = {r2:.3f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel('Actual PBL Height (m)')
            ax.set_ylabel('Predicted PBL Height (m)')
            ax.set_title(f'{label} Condition')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    individual_path = os.path.join(output_dir, 'scatter_by_condition.png')
    plt.savefig(individual_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Individual scatter plots saved to: {individual_path}")

    # Plot 3: Density scatter plots
    density_width_cm = 18.0
    density_height_cm = 14.0
    fig_width_in = density_width_cm / 2.54
    fig_height_in = density_height_cm / 2.54
    fig, axes = plt.subplots(2, 2, figsize=(fig_width_in, fig_height_in))

    for idx, (condition_name, (cond_targets, cond_preds, _, color, label)) in enumerate(conditions_data.items()):
        if len(cond_targets) > 0:
            ax = axes[idx // 2, idx % 2]

            # Create 2D histogram
            hb = ax.hexbin(cond_targets, cond_preds, gridsize=50, cmap='turbo',
                           bins='log', mincnt=1)

            # Add colorbar
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label('log10(count)', fontsize=10.5)
            cb.ax.tick_params(labelsize=10.5)

            # Add perfect prediction line
            cond_min = min(cond_targets.min(), cond_preds.min())
            cond_max = max(cond_targets.max(), cond_preds.max())
            ax.plot([cond_min, cond_max], [cond_min, cond_max], 'r--', alpha=0.7, linewidth=1.5)

            # Add 1:1 line
            ax.plot([cond_min, cond_max], [cond_min, cond_max], 'k-', alpha=0.3, linewidth=0.5)

            # Add statistics
            rmse = np.sqrt(mean_squared_error(cond_targets, cond_preds))
            r2 = r2_score(cond_targets, cond_preds) if len(cond_preds) > 1 else 0

            stats_text = f'n = {len(cond_targets)}\nRMSE = {rmse:.1f} m\nR² = {r2:.3f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=10.5,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlabel('Actual PBL Height (m)')
            ax.set_ylabel('Predicted PBL Height (m)')
            ax.set_title(f'{label} Condition (Density)')
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    density_path = os.path.join(output_dir, 'scatter_density.png')
    plt.savefig(density_path, dpi=300)
    plt.close()
    print(f"Density scatter plots saved to: {density_path}")

    # Plot 4: Error distribution by condition
    plt.figure(figsize=(12, 8))

    errors = preds - targets
    clear_errors = errors[clear_mask]
    cloudy_errors = errors[cloudy_mask]
    fog_errors = errors[fog_mask]

    error_data = [errors, clear_errors, cloudy_errors, fog_errors]
    labels = ['All', 'Clear', 'Cloudy', 'Fog/Mist']
    colors = ['gray', 'blue', 'green', 'red']

    # Create violin plot
    parts = plt.violinplot(error_data, showmeans=True, showmedians=True)

    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)

    # Customize
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel('Prediction Error (m)')
    plt.title('Error Distribution by Condition')
    plt.grid(True, alpha=0.3, axis='y')

    # Add horizontal line at 0
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)

    # Add statistics
    for i, err_data in enumerate(error_data):
        if len(err_data) > 0:
            mean_err = np.mean(err_data)
            std_err = np.std(err_data)
            median_err = np.median(err_data)

            # Add text box
            stats_text = f'Mean: {mean_err:.1f} m\nStd: {std_err:.1f} m\nMed: {median_err:.1f} m'
            plt.text(i + 0.8, plt.ylim()[1] * 0.9, stats_text, fontsize=8,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    error_path = os.path.join(output_dir, 'error_distribution.png')
    plt.savefig(error_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Error distribution plot saved to: {error_path}")


def plot_evaluation_results(preds, targets, cloud_status, output_dir):
    """Plot evaluation results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Scatter plot
    axes[0, 0].scatter(targets, preds, alpha=0.5, s=1)
    axes[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()],
                    'r--', alpha=0.5)
    axes[0, 0].set_xlabel('Actual (m)')
    axes[0, 0].set_ylabel('Predicted (m)')
    axes[0, 0].set_title('Predicted vs Actual')
    axes[0, 0].grid(True, alpha=0.3)

    # Error distribution
    errors = preds - targets
    axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Error (m)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Error Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # Error by condition
    condition_errors = []
    condition_names = []
    condition_masks = [
        cloud_status == 0,
        (cloud_status >= 1) & (cloud_status <= 3),
        cloud_status >= 4
    ]

    for mask, name in zip(condition_masks, ['Clear', 'Cloudy', 'Fog/Mist']):
        if np.sum(mask) > 0:
            cond_errors = errors[mask]
            condition_errors.append(cond_errors)
            condition_names.append(name)

    if condition_errors:
        axes[0, 2].boxplot(condition_errors, labels=condition_names)
        axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 2].set_ylabel('Error (m)')
        axes[0, 2].set_title('Error Distribution by Condition')
        axes[0, 2].grid(True, alpha=0.3)

    # Time series example
    n_plot = min(500, len(preds))
    axes[1, 0].plot(range(n_plot), preds[:n_plot], 'b-', alpha=0.7, label='Predicted')
    axes[1, 0].plot(range(n_plot), targets[:n_plot], 'r-', alpha=0.7, label='Actual')
    axes[1, 0].fill_between(range(n_plot),
                            preds[:n_plot] - errors[:n_plot],
                            preds[:n_plot] + errors[:n_plot],
                            alpha=0.2, color='gray')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('PBL Height (m)')
    axes[1, 0].set_title('Prediction Sequence Example')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Error vs Height
    axes[1, 1].scatter(targets, np.abs(errors), alpha=0.5, s=1)
    axes[1, 1].set_xlabel('Actual Height (m)')
    axes[1, 1].set_ylabel('Absolute Error (m)')
    axes[1, 1].set_title('Error vs Height')
    axes[1, 1].grid(True, alpha=0.3)

    # Cumulative distribution function
    sorted_errors = np.sort(np.abs(errors))
    y_vals = np.arange(len(sorted_errors)) / float(len(sorted_errors))
    axes[1, 2].plot(sorted_errors, y_vals, 'b-')
    axes[1, 2].set_xlabel('Absolute Error (m)')
    axes[1, 2].set_ylabel('Cumulative Probability')
    axes[1, 2].set_title('Error Cumulative Distribution')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'evaluation_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Evaluation plots saved to: {plot_path}")


def plot_training_history(history, output_dir):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curve
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # RMSE curve
    axes[0, 1].plot(history['train_rmse'], label='Train RMSE')
    axes[0, 1].plot(history['val_rmse'], label='Val RMSE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('RMSE (m)')
    axes[0, 1].set_title('Training and Validation RMSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # MAE curve
    axes[1, 0].plot(history['train_mae'], label='Train MAE')
    axes[1, 0].plot(history['val_mae'], label='Val MAE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE (m)')
    axes[1, 0].set_title('Training and Validation MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate curve
    axes[1, 1].plot(history['learning_rate'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    history_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(history_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training history plot saved to: {history_path}")
