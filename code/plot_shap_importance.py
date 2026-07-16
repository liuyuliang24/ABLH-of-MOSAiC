"""
Redraw SHAP group-importance plots from an existing CSV result file.

This script only handles plotting. It does not recompute SHAP values.
"""
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

matplotlib.use('Agg')


DISPLAY_LABELS = {
    'wind_speed_shear': 'Wind Shear',
    'cbh_cloud_status': 'CBH + Cloud Status',
    'avg_wind_speed': 'Mean Wind Speed',
    'infrared_raw': 'AERI Raw',
    'time_features': 'Time Features',
    'wind_raw': 'Wind Raw',
    'others': 'Others',
    'kv_contrast': 'K-V Contrast',
    'v_band_raw': 'V-band Raw',
    'high_freq_window_raw': 'Window Raw',
    'k_band_raw': 'K-band Raw',
    'ceilometer_raw': 'Ceilometer Raw',
    'backscatter_5layer': 'Backscatter 5-Layer',
    'infrared_derived': 'AERI Derived',
    'g_band_raw': '183 GHz Raw',
    'window_region': 'Window Derived',
    'band_183_slope': '183 GHz Slope',
    'backscatter_peak_transition': 'Peak + Transition',
    'k_band_slope': 'K-band Slope',
    'v_band_slope': 'V-band Slope',
    'miracp_183_combined': '183 GHz Combined',
    'miracp_window_combined': 'Window Region',
}


PLOT_GROUP_MAP = {
    'g_band_raw': 'miracp_183_combined',
    'band_183_slope': 'miracp_183_combined',
    'high_freq_window_raw': 'miracp_window_combined',
    'window_region': 'miracp_window_combined',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Redraw SHAP importance plots from CSV.')
    parser.add_argument(
        '--csv',
        default='pbl_results_fixed_v3/shap_analysis/shap_group_importance.csv',
        help='Path to shap_group_importance.csv',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Directory for rewritten figures. Default: same directory as CSV.',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=12,
        help='Number of top groups to show per panel.',
    )
    return parser.parse_args()


def prettify_group_name(group_name):
    return DISPLAY_LABELS.get(group_name, group_name.replace('_', ' ').title())


def prepare_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    if 'condition' not in df.columns or 'group' not in df.columns:
        raise ValueError(f'Unexpected CSV format: {csv_path}')
    df['plot_group'] = df['group'].map(lambda x: PLOT_GROUP_MAP.get(x, x))

    aggregated = (
        df.groupby(['condition', 'plot_group', 'source_family', 'color', 'sample_count'], as_index=False)
        .agg({
            'importance': 'sum',
            'importance_ratio_percent': 'sum',
            'feature_count': 'sum',
        })
    )
    aggregated['display_group'] = aggregated['plot_group'].map(prettify_group_name)
    return aggregated


def build_legend_handles(df):
    legend_df = df[['source_family', 'color']].drop_duplicates().sort_values('source_family')
    handles = [
        Patch(facecolor=row['color'], edgecolor='none', label=row['source_family'])
        for _, row in legend_df.iterrows()
    ]
    return handles


def plot_overall(df, output_dir, prefix, top_k):
    overall = df[df['condition'] == 'Overall'].sort_values('importance', ascending=False).head(top_k).copy()
    overall = overall.iloc[::-1]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(overall['display_group'], overall['importance'], color=overall['color'])
    ax.set_xlabel('Importance')
    ax.set_ylabel('')
    ax.set_title(f'Overall SHAP Group Importance (Top {len(overall)})')
    ax.grid(True, axis='x', alpha=0.25)

    for _, row in overall.iterrows():
        ax.text(row['importance'], row['display_group'], f"  {row['importance']:.2f}",
                va='center', ha='left', fontsize=9)

    handles = build_legend_handles(df)
    fig.legend(handles=handles, loc='upper center', ncol=min(len(handles), 4), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_path = os.path.join(output_dir, f'{prefix}_overall_clean.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_by_condition(df, output_dir, prefix, top_k):
    condition_order = ['Overall', 'Clear', 'Cloudy', 'Fog-Mist']
    overall_rank = (
        df[df['condition'] == 'Overall']
        .sort_values('importance', ascending=False)['plot_group']
        .tolist()
    )
    rank_map = {name: idx for idx, name in enumerate(overall_rank)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, condition in zip(axes, condition_order):
        cond_df = df[df['condition'] == condition].copy()
        if cond_df.empty:
            ax.axis('off')
            continue

        cond_df['rank'] = cond_df['plot_group'].map(rank_map).fillna(10**9)
        cond_df = cond_df.sort_values(['importance', 'rank'], ascending=[False, True]).head(top_k)
        cond_df = cond_df.sort_values('importance', ascending=True)

        ax.barh(cond_df['display_group'], cond_df['importance'], color=cond_df['color'])
        sample_count = int(cond_df['sample_count'].iloc[0])
        ax.set_title(f'{condition} ')
        ax.set_xlabel('Importance')
        ax.set_ylabel('')
        ax.grid(True, axis='x', alpha=0.25)

        for _, row in cond_df.iterrows():
            ax.text(row['importance'], row['display_group'], f"  {row['importance']:.2f}",
                    va='center', ha='left', fontsize=8)

    handles = build_legend_handles(df)
    fig.legend(handles=handles, loc='upper center', ncol=min(len(handles), 4), frameon=False)
    fig.suptitle('SHAP Group Importance by Weather Condition', fontsize=18, y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out_path = os.path.join(output_dir, f'{prefix}_by_condition_clean.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    df = prepare_dataframe(args.csv)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(output_dir, exist_ok=True)

    prefix = os.path.splitext(os.path.basename(args.csv))[0]
    overall_path = plot_overall(df, output_dir, prefix, args.top_k)
    condition_path = plot_by_condition(df, output_dir, prefix, args.top_k)

    print('Saved files:')
    print(f'  {overall_path}')
    print(f'  {condition_path}')


if __name__ == '__main__':
    main()
