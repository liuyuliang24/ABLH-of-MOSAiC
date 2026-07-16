"""
Deterministic full-dataset SHAP analysis for the trained PBLH model.

Design choices:
1. Use the entire currently configured dataset file via mode='all'.
2. Use SHAP only. If shap is unavailable, raise an error instead of silently
   falling back to another method.
3. Use DeepExplainer with a deterministic background subset for speed and
   reproducibility.
4. Accumulate group importance batch by batch across the full dataset, so the
   final score uses all samples instead of a small random subset.
"""
import argparse
import os
from collections import OrderedDict, defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use('Agg')

from config import TRAIN_CONFIG, MODEL_CONFIG
from dataset import PBLDatasetFixed
from models import create_pbl_model
from plot_font_utils import get_available_serif_font
from utils import create_output_dir, get_device, safe_load_checkpoint


CONDITION_NAME_MAP = {
    0: 'Clear',
    1: 'Cloudy',
    2: 'Fog-Mist',
}

SOURCE_COLORS = {
    'HATPRO': '#1f77b4',
    'MiRAC-P': '#ff7f0e',
    'AERI': '#2ca02c',
    'Wind': '#d62728',
    'Ceilometer': '#9467bd',
    'CloudBase/Status': '#8c564b',
    'Time': '#e377c2',
    'Other': '#7f7f7f',
}

DISPLAY_LABELS = {
    'wind_raw': 'wind',
    'infrared_raw': 'infrared',
    'g_band_raw': 'G-band',
    'cbh_cloud_status': 'CBH',
    'ceilometer_raw': 'Ceil',
    'k_band_raw': 'K-band',
    'wind_speed_shear': 'wind-shar',
    'v_band_raw': 'v-band',
    'avg_wind_speed': 'wind-avg',
    'high_freq_window_raw': 'window',
    'others': 'others',
    'time_features': 'time',
}

FONT_SIZE_PT = 10.5
PANEL_WIDTH_CM = 18.0
PANEL_HEIGHT_CM = PANEL_WIDTH_CM * 12.0 / 16.0


def _set_plot_style():
    serif_font = get_available_serif_font()
    plt.rcParams.update(
        {
            'font.family': serif_font,
            'font.size': FONT_SIZE_PT,
            'axes.titlesize': FONT_SIZE_PT,
            'axes.labelsize': FONT_SIZE_PT,
            'xtick.labelsize': FONT_SIZE_PT,
            'ytick.labelsize': FONT_SIZE_PT,
        }
    )


def _display_group_name(group_name):
    return DISPLAY_LABELS.get(group_name, group_name)


def parse_args():
    parser = argparse.ArgumentParser(description='Full-dataset deterministic SHAP analysis for PBLH model.')
    parser.add_argument('--mode', choices=['all'], default='all',
                        help='Use the entire configured dataset file without any train/val split.')
    parser.add_argument('--method', choices=['shap'], default='shap',
                        help='Only SHAP is supported in this script.')
    parser.add_argument('--background-size', type=int, default=128,
                        help='Deterministic background sample count for DeepExplainer.')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size used for SHAP evaluation over the full dataset.')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory. Default: <TRAIN_CONFIG[output_dir]>/shap_analysis')
    parser.add_argument('--device', default=None,
                        help='Override device, e.g. cpu or cuda.')
    return parser.parse_args()


def resolve_output_dir(user_output_dir):
    if user_output_dir:
        return user_output_dir
    return os.path.join(TRAIN_CONFIG['output_dir'], 'shap_analysis')


def condition_labels_from_status(cloud_status):
    cloud_status = np.asarray(cloud_status)
    return np.where(
        cloud_status == 0,
        0,
        np.where(cloud_status >= 4, 2, 1)
    ).astype(np.int64)


def evenly_spaced_pick(indices, count):
    indices = np.asarray(indices, dtype=np.int64)
    if count <= 0 or len(indices) == 0:
        return np.array([], dtype=np.int64)
    if count >= len(indices):
        return indices.copy()
    positions = np.linspace(0, len(indices) - 1, num=count, dtype=int)
    return indices[positions]


def deterministic_background_indices(condition_labels, background_size):
    total_count = len(condition_labels)
    if background_size >= total_count:
        return np.arange(total_count, dtype=np.int64)

    unique_conditions = [cond for cond in sorted(CONDITION_NAME_MAP.keys()) if np.sum(condition_labels == cond) > 0]
    base_count = background_size // len(unique_conditions)
    remainder = background_size % len(unique_conditions)

    chosen = []
    remaining_pool = []
    for rank, cond in enumerate(unique_conditions):
        cond_indices = np.where(condition_labels == cond)[0]
        take_n = min(len(cond_indices), base_count + (1 if rank < remainder else 0))
        picked = evenly_spaced_pick(cond_indices, take_n)
        chosen.extend(picked.tolist())

        picked_set = set(picked.tolist())
        remaining_pool.extend([idx for idx in cond_indices.tolist() if idx not in picked_set])

    if len(chosen) < background_size:
        need = background_size - len(chosen)
        extra = evenly_spaced_pick(np.array(sorted(remaining_pool), dtype=np.int64), need)
        chosen.extend(extra.tolist())

    return np.array(sorted(chosen), dtype=np.int64)


def load_training_scalers():
    scaler_path = os.path.join(TRAIN_CONFIG['output_dir'], TRAIN_CONFIG['scaler_save_path'])
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f'Scaler file not found: {scaler_path}')
    return joblib.load(scaler_path)


def load_dataset(external_scalers):
    use_aeri = TRAIN_CONFIG.get('use_aeri', True)
    use_microwave = TRAIN_CONFIG.get('use_microwave', True)
    use_wind_features = TRAIN_CONFIG.get('use_wind_features', True)
    use_ceilometer = TRAIN_CONFIG.get('use_ceilometer', True)
    return PBLDatasetFixed(
        features_file=TRAIN_CONFIG['features_file1'],
        labels_file=TRAIN_CONFIG['labels_file1'],
        mode='all',
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
        aeri_channels=TRAIN_CONFIG.get('aeri_channels', 0) if use_aeri else 0,
        use_aeri=use_aeri,
        use_microwave=use_microwave,
        use_ceilometer=use_ceilometer,
    )


def load_model(dataset, device):
    model_path = os.path.join(TRAIN_CONFIG['output_dir'], TRAIN_CONFIG['model_save_path'])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found: {model_path}')

    checkpoint = safe_load_checkpoint(model_path, weights_only=False)
    if checkpoint is None:
        raise RuntimeError(f'Failed to load checkpoint: {model_path}')

    actual_physics_dim = dataset.physics_features.shape[1]
    actual_hatpro_dim = dataset.hatpro.shape[1]
    actual_miracp_dim = dataset.miracp.shape[1]
    actual_wind_dim = dataset.u_wind.shape[1]
    actual_aeri_dim = dataset.aeri_channels
    actual_cloud_classes = int(dataset.cloud_status.max() + 1)

    model = create_pbl_model(
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


class DifferentiablePBLWrapper(nn.Module):
    """
    SHAP wrapper around the trained model.

    cloud_status is represented as a float one-hot vector so DeepExplainer can
    propagate attributions through the weather embeddings. On actual samples the
    one-hot vectors are exact; only the background interpolation inside SHAP
    becomes continuous.
    """

    def __init__(self, base_model, input_names):
        super().__init__()
        self.base_model = base_model
        self.input_names = list(input_names)

    def forward(self, *inputs):
        input_map = dict(zip(self.input_names, inputs))
        cloud_onehot = input_map['cloud_onehot']
        batch_size = cloud_onehot.size(0)
        ref = input_map.get('backscatter', cloud_onehot)

        hatpro = input_map.get('hatpro')
        if hatpro is None:
            hatpro = ref.new_zeros(batch_size, 0)

        miracp = input_map.get('miracp')
        if miracp is None:
            miracp = ref.new_zeros(batch_size, 0)

        aeri_rad = input_map.get('aeri_rad')
        backscatter = input_map['backscatter']
        u_wind = input_map['u_wind']
        v_wind = input_map['v_wind']
        physics = input_map['physics']

        cloud_onehot = torch.clamp(cloud_onehot, min=0.0)
        cloud_norm = cloud_onehot.sum(dim=1, keepdim=True).clamp_min(1e-6)
        cloud_probs = cloud_onehot / cloud_norm

        clear_prob = cloud_probs[:, 0:1]
        cloudy_prob = cloud_probs[:, 1:4].sum(dim=1, keepdim=True)
        fog_prob = cloud_probs[:, 4:].sum(dim=1, keepdim=True)
        condition_probs = torch.cat([clear_prob, cloudy_prob, fog_prob], dim=1)

        cloud_feat = cloud_probs @ self.base_model.cloud_embed.weight
        condition_feat = condition_probs @ self.base_model.condition_embed.weight
        weather_context = torch.cat([cloud_feat, condition_feat], dim=1)

        hatpro_features = self.base_model.hatpro_encoder(
            hatpro, self.base_model._build_spectral_channels
        )
        miracp_features = self.base_model.miracp_encoder(
            miracp, self.base_model._build_spectral_channels
        )

        if self.base_model.aeri_encoder is not None and aeri_rad is not None and aeri_rad.numel() > 0:
            aeri_features = self.base_model.aeri_encoder(aeri_rad)
        else:
            aeri_features = hatpro.new_zeros(batch_size, self.base_model.aeri_out_dim)

        physics_core = physics[:, :self.base_model.physics_core_dim]
        physics_cbh = physics[:, self.base_model.physics_core_dim:self.base_model.physics_core_dim + 2]
        physics_time = physics[:, -3:]

        physics_core_feat = self.base_model.physics_core_branch(physics_core)
        cbh_feat = self.base_model.cbh_branch(physics_cbh)
        temporal_feat = self.base_model.temporal_branch(physics_time)

        bs_in = self.base_model._build_profile_channels(backscatter)
        ceil_avg_feat = self.base_model.ceil_conv(bs_in)
        ceil_peak_feat = self.base_model.ceil_peak_encoder(bs_in)
        near_surface_feat = self.base_model.near_surface_conv(
            bs_in[:, :, :self.base_model.near_surface_bins]
        )

        if self.base_model.use_wind_branch and u_wind is not None and v_wind is not None:
            wind_features = self.base_model.wind_conv(
                self.base_model._build_wind_channels(u_wind, v_wind)
            )
        else:
            wind_features = hatpro.new_zeros(batch_size, self.base_model.wind_out_dim)

        features = torch.cat([
            hatpro_features,
            miracp_features,
            aeri_features,
            physics_core_feat,
            cbh_feat,
            temporal_feat,
            ceil_avg_feat,
            ceil_peak_feat,
            near_surface_feat,
            wind_features,
            cloud_feat,
            condition_feat,
        ], dim=1)

        fused = self.base_model.fusion_proj(features)
        film_gamma, film_beta = self.base_model.condition_film(weather_context).chunk(2, dim=1)
        fused = fused * (1.0 + 0.12 * torch.tanh(film_gamma)) + 0.12 * film_beta
        fused = self.base_model.fusion_res1(fused)
        fused = self.base_model.fusion_res2(fused)

        shared = self.base_model.shared_head(fused)
        gate_logits = self.base_model.expert_gate(torch.cat([shared, weather_context], dim=1))
        gate = torch.softmax(
            gate_logits + self.base_model.condition_prior_scale * condition_probs, dim=1
        )

        expert_features = torch.stack(
            [expert(shared) for expert in self.base_model.expert_heads], dim=1
        )
        expert_mix = (gate.unsqueeze(-1) * expert_features).sum(dim=1)

        output_features = shared + 0.28 * expert_mix
        output_features = self.base_model.output_res(output_features)
        out = self.base_model.output_head(output_features)

        shallow_gate = self.base_model.shallow_regime_gate(
            torch.cat([near_surface_feat, cbh_feat, temporal_feat, condition_feat], dim=1)
        )
        shallow_delta = self.base_model.shallow_residual(
            torch.cat([output_features, near_surface_feat, cbh_feat, temporal_feat, condition_feat], dim=1)
        )
        out = out + 0.16 * shallow_gate * shallow_delta

        cloudy_context = torch.cat(
            [near_surface_feat, ceil_peak_feat, cbh_feat, temporal_feat, wind_features, cloud_feat, condition_feat],
            dim=1
        )
        cloudy_gate = self.base_model.cloudy_refine_gate(cloudy_context)
        cloudy_delta = self.base_model.cloudy_refine_head(
            torch.cat([output_features, cloudy_context], dim=1)
        )
        out = out + self.base_model.cloudy_refine_scale * cloudy_prob * cloudy_gate * cloudy_delta

        return out


def make_input_arrays(dataset, model):
    cloud_classes = model.cloud_embed.num_embeddings
    cloud_onehot = np.eye(cloud_classes, dtype=np.float32)[dataset.cloud_status.astype(np.int64)]

    arrays = OrderedDict()
    if dataset.hatpro.shape[1] > 0:
        arrays['hatpro'] = dataset.hatpro.astype(np.float32)
    if dataset.miracp.shape[1] > 0:
        arrays['miracp'] = dataset.miracp.astype(np.float32)
    arrays['backscatter'] = dataset.backscatter.astype(np.float32)
    arrays['u_wind'] = dataset.u_wind.astype(np.float32)
    arrays['v_wind'] = dataset.v_wind.astype(np.float32)
    arrays['physics'] = dataset.physics_features.astype(np.float32)
    if dataset.aeri_channels > 0:
        arrays['aeri_rad'] = dataset.aeri_rad.astype(np.float32)
    arrays['cloud_onehot'] = cloud_onehot
    return arrays


def slice_arrays(arrays, indices):
    return OrderedDict((name, value[indices]) for name, value in arrays.items())


def arrays_to_tensors(arrays, device):
    return [torch.tensor(value, dtype=torch.float32, device=device) for value in arrays.values()]


def get_group_source(group_name):
    if group_name.startswith('k_band') or group_name.startswith('v_band') or group_name == 'kv_contrast':
        return 'HATPRO'
    if group_name.startswith('g_band') or group_name.startswith('high_freq_window') or group_name.startswith('band_183') or group_name == 'window_region':
        return 'MiRAC-P'
    if group_name.startswith('infrared'):
        return 'AERI'
    if group_name.startswith('wind') or group_name == 'avg_wind_speed':
        return 'Wind'
    if group_name.startswith('ceilometer') or group_name.startswith('backscatter'):
        return 'Ceilometer'
    if group_name.startswith('cbh'):
        return 'CloudBase/Status'
    if group_name.startswith('time'):
        return 'Time'
    return 'Other'


def build_group_definitions(dataset, model, input_names):
    hatpro_k_dim = getattr(model.hatpro_encoder, 'k_band_dim', min(7, dataset.hatpro.shape[1]))
    hatpro_v_dim = getattr(model.hatpro_encoder, 'v_band_dim', max(dataset.hatpro.shape[1] - hatpro_k_dim, 0))
    miracp_g_dim = getattr(model.miracp_encoder, 'k_band_dim', min(6, dataset.miracp.shape[1]))
    miracp_window_dim = getattr(model.miracp_encoder, 'v_band_dim', max(dataset.miracp.shape[1] - miracp_g_dim, 0))

    input_index = {name: idx for idx, name in enumerate(input_names)}
    groups = OrderedDict()

    def add_group(name, refs):
        groups[name] = {
            'refs': refs,
            'source_family': get_group_source(name),
        }

    if 'hatpro' in input_index:
        add_group('k_band_raw', [('hatpro', np.arange(0, hatpro_k_dim, dtype=np.int64))])
        add_group('v_band_raw', [('hatpro', np.arange(hatpro_k_dim, hatpro_k_dim + hatpro_v_dim, dtype=np.int64))])
    if 'miracp' in input_index:
        add_group('g_band_raw', [('miracp', np.arange(0, miracp_g_dim, dtype=np.int64))])
        add_group('high_freq_window_raw', [('miracp', np.arange(miracp_g_dim, miracp_g_dim + miracp_window_dim, dtype=np.int64))])
    if 'aeri_rad' in input_index:
        add_group('infrared_raw', [('aeri_rad', np.arange(dataset.aeri_rad.shape[1], dtype=np.int64))])

    add_group('wind_raw', [
        ('u_wind', np.arange(dataset.u_wind.shape[1], dtype=np.int64)),
        ('v_wind', np.arange(dataset.v_wind.shape[1], dtype=np.int64)),
    ])
    add_group('ceilometer_raw', [('backscatter', np.arange(dataset.backscatter.shape[1], dtype=np.int64))])

    used_physics = set()
    offset = 0
    if dataset.use_wind_features:
        add_group('avg_wind_speed', [('physics', np.arange(offset, offset + 3, dtype=np.int64))])
        used_physics.update(range(offset, offset + 3))
        add_group('wind_speed_shear', [('physics', np.arange(offset + 3, offset + 8, dtype=np.int64))])
        used_physics.update(range(offset + 3, offset + 8))
        offset += 8

    add_group('backscatter_5layer', [('physics', np.array([offset + 0, offset + 1, offset + 2, offset + 3, offset + 6], dtype=np.int64))])
    used_physics.update([offset + 0, offset + 1, offset + 2, offset + 3, offset + 6])
    add_group('backscatter_peak_transition', [('physics', np.array([offset + 4, offset + 5], dtype=np.int64))])
    used_physics.update([offset + 4, offset + 5])
    offset += 7

    if 'hatpro' in input_index:
        add_group('k_band_slope', [('physics', np.array([offset + 2], dtype=np.int64))])
        used_physics.add(offset + 2)
        add_group('v_band_slope', [('physics', np.array([offset + 3], dtype=np.int64))])
        used_physics.add(offset + 3)
        add_group('kv_contrast', [('physics', np.array([offset + 4], dtype=np.int64))])
        used_physics.add(offset + 4)
    offset += 5

    if 'miracp' in input_index:
        add_group('band_183_slope', [('physics', np.array([offset + 2], dtype=np.int64))])
        used_physics.add(offset + 2)
        add_group('window_region', [('physics', np.array([offset + 1, offset + 3], dtype=np.int64))])
        used_physics.update([offset + 1, offset + 3])
    offset += 4

    if 'aeri_rad' in input_index:
        add_group('infrared_derived', [('physics', np.arange(offset, offset + 5, dtype=np.int64))])
        used_physics.update(range(offset, offset + 5))
    offset += 5

    add_group('cbh_cloud_status', [
        ('physics', np.array([offset + 0, offset + 1], dtype=np.int64)),
        ('cloud_onehot', np.arange(model.cloud_embed.num_embeddings, dtype=np.int64)),
    ])
    used_physics.update([offset + 0, offset + 1])
    add_group('time_features', [('physics', np.arange(offset + 2, offset + 5, dtype=np.int64))])
    used_physics.update(range(offset + 2, offset + 5))

    remaining_physics = np.array([idx for idx in range(dataset.physics_features.shape[1]) if idx not in used_physics], dtype=np.int64)
    if len(remaining_physics) > 0:
        add_group('others', [('physics', remaining_physics)])

    for meta in groups.values():
        meta['color'] = SOURCE_COLORS[meta['source_family']]
        meta['feature_count'] = int(sum(len(indices) for _, indices in meta['refs']))

    return groups


def import_shap():
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            'shap is required for this script. Install it first, for example: python -m pip install shap'
        ) from exc
    return shap


def standardize_shap_array(array):
    array = np.asarray(array)
    if array.ndim == 1:
        return array.reshape(array.shape[0], 1)
    return array.reshape(array.shape[0], -1)


def compute_batch_group_scores(shap_by_input, group_defs):
    batch_size = next(iter(shap_by_input.values())).shape[0]
    group_scores = {}
    for group_name, meta in group_defs.items():
        values = np.zeros(batch_size, dtype=np.float64)
        for input_name, indices in meta['refs']:
            arr = shap_by_input[input_name]
            values += np.abs(arr[:, indices]).sum(axis=1)
        group_scores[group_name] = values
    return group_scores


def compute_condition_results(explainer, eval_arrays, condition_labels, group_defs, input_names, batch_size, device):
    accum = {
        'Overall': defaultdict(float),
        'Clear': defaultdict(float),
        'Cloudy': defaultdict(float),
        'Fog-Mist': defaultdict(float),
    }
    sample_counts = {
        'Overall': 0,
        'Clear': 0,
        'Cloudy': 0,
        'Fog-Mist': 0,
    }

    total_samples = len(condition_labels)
    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        batch_arrays = OrderedDict((name, value[start:end]) for name, value in eval_arrays.items())
        batch_tensors = arrays_to_tensors(batch_arrays, device)
        batch_shap = explainer.shap_values(batch_tensors, check_additivity=False)

        if isinstance(batch_shap, list) and len(batch_shap) == 1 and isinstance(batch_shap[0], list):
            batch_shap = batch_shap[0]
        if not isinstance(batch_shap, list):
            raise RuntimeError('Unexpected SHAP output format for multi-input model.')

        shap_by_input = OrderedDict()
        for input_name, shap_array in zip(input_names, batch_shap):
            shap_by_input[input_name] = standardize_shap_array(shap_array)

        batch_group_scores = compute_batch_group_scores(shap_by_input, group_defs)
        batch_conditions = condition_labels[start:end]

        for group_name, group_values in batch_group_scores.items():
            accum['Overall'][group_name] += float(group_values.sum())
            for cond_id, cond_name in CONDITION_NAME_MAP.items():
                mask = batch_conditions == cond_id
                if np.any(mask):
                    accum[cond_name][group_name] += float(group_values[mask].sum())

        sample_counts['Overall'] += len(batch_conditions)
        for cond_id, cond_name in CONDITION_NAME_MAP.items():
            sample_counts[cond_name] += int(np.sum(batch_conditions == cond_id))

        print(f'Processed SHAP batch: {end}/{total_samples}')

    condition_results = OrderedDict()
    for condition_name in ['Overall', 'Clear', 'Cloudy', 'Fog-Mist']:
        if sample_counts[condition_name] == 0:
            continue
        group_scores = {
            group_name: accum[condition_name][group_name] / sample_counts[condition_name]
            for group_name in group_defs.keys()
        }
        condition_results[condition_name] = {
            'group_scores': group_scores,
            'sample_count': sample_counts[condition_name],
        }
    return condition_results


def result_frame_from_scores(condition_results, group_defs):
    rows = []
    for condition_name, meta in condition_results.items():
        total = float(sum(meta['group_scores'].values())) or 1.0
        for group_name, importance in meta['group_scores'].items():
            group_meta = group_defs[group_name]
            rows.append({
                'condition': condition_name,
                'group': group_name,
                'source_family': group_meta['source_family'],
                'color': group_meta['color'],
                'importance': importance,
                'importance_ratio_percent': importance / total * 100.0,
                'feature_count': group_meta['feature_count'],
                'sample_count': meta['sample_count'],
            })
    df = pd.DataFrame(rows)
    return df.sort_values(['condition', 'importance'], ascending=[True, False]).reset_index(drop=True)


def plot_condition_panels(result_df, panel_fig_path):
    _set_plot_style()
    ordered_conditions = ['Overall', 'Clear', 'Cloudy', 'Fog-Mist']
    overall_df = result_df[result_df['condition'] == 'Overall'].sort_values('importance', ascending=False).head(12)
    overall_order = overall_df['group'].tolist()
    fig_width_in = PANEL_WIDTH_CM / 2.54
    fig_height_in = PANEL_HEIGHT_CM / 2.54
    fig, axes = plt.subplots(2, 2, figsize=(fig_width_in, fig_height_in))
    axes = axes.flatten()
    for ax, condition_name in zip(axes, ordered_conditions):
        cond_df = result_df[result_df['condition'] == condition_name].copy()
        if cond_df.empty:
            ax.axis('off')
            continue
        cond_df['order'] = cond_df['group'].apply(
            lambda x: overall_order.index(x) if x in overall_order else len(overall_order)
        )
        cond_df = cond_df.sort_values(['order', 'importance'], ascending=[True, False]).head(12)
        cond_df = cond_df.sort_values('importance', ascending=True)
        display_labels = cond_df['group'].map(_display_group_name)
        ax.barh(display_labels, cond_df['importance'], color=cond_df['color'])
        ax.set_title(f'{condition_name} (n={int(cond_df["sample_count"].iloc[0])})', fontsize=FONT_SIZE_PT)
        ax.set_xlabel('Importance', fontsize=FONT_SIZE_PT)
        ax.set_ylabel('')
        ax.tick_params(axis='both', labelsize=FONT_SIZE_PT)
        ax.grid(True, axis='x', alpha=0.25)
    plt.tight_layout()
    fig.savefig(panel_fig_path, dpi=300)
    plt.close(fig)


def save_outputs(result_df, output_dir, condition_results, group_defs, background_indices):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'shap_group_importance.csv')
    result_df.to_csv(csv_path, index=False)

    npz_path = os.path.join(output_dir, 'shap_raw_importance.npz')
    payload = {
        'background_indices': np.array(background_indices, dtype=np.int64),
        'group_names': np.array(list(group_defs.keys()), dtype=object),
        'group_feature_counts': np.array([meta['feature_count'] for meta in group_defs.values()], dtype=np.int64),
        'condition_names': np.array(list(condition_results.keys()), dtype=object),
    }
    for condition_name, meta in condition_results.items():
        safe_name = condition_name.lower().replace('-', '_')
        payload[f'{safe_name}_group_scores'] = np.array(
            [meta['group_scores'][group_name] for group_name in group_defs.keys()],
            dtype=np.float64
        )
        payload[f'{safe_name}_sample_count'] = np.array([meta['sample_count']], dtype=np.int64)
    np.savez(npz_path, **payload)

    txt_path = os.path.join(output_dir, 'shap_group_importance.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('Deep SHAP grouped importance over the full configured dataset\n')
        f.write('=' * 72 + '\n')
        for condition_name in ['Overall', 'Clear', 'Cloudy', 'Fog-Mist']:
            cond_df = result_df[result_df['condition'] == condition_name]
            if len(cond_df) == 0:
                continue
            f.write(f'\n[{condition_name}] samples={int(cond_df["sample_count"].iloc[0])}\n')
            for _, row in cond_df.iterrows():
                f.write(
                    f"{row['group']} ({row['source_family']}): importance={row['importance']:.6f}, "
                    f"ratio={row['importance_ratio_percent']:.2f}%, features={int(row['feature_count'])}\n"
                )

    # Quick built-in plots; cleaner redrawing can still be done by plot_shap_importance.py.
    _set_plot_style()
    overall_df = result_df[result_df['condition'] == 'Overall'].sort_values('importance', ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(12, 6))
    overall_labels = overall_df['group'].map(_display_group_name)
    ax.barh(overall_labels[::-1], overall_df['importance'][::-1], color=overall_df['color'][::-1])
    ax.set_xlabel('Importance', fontsize=FONT_SIZE_PT)
    ax.set_title('Deep SHAP Group Importance (Overall, Top 12)', fontsize=FONT_SIZE_PT)
    ax.grid(True, axis='x', alpha=0.25)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'shap_group_importance.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    panel_fig_path = os.path.join(output_dir, 'shap_group_importance_by_condition.png')
    plot_condition_panels(result_df, panel_fig_path)

    return csv_path, npz_path, txt_path, fig_path, panel_fig_path


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else get_device()
    TRAIN_CONFIG['device'] = str(device)
    create_output_dir(TRAIN_CONFIG)

    output_dir = resolve_output_dir(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 72)
    print('Full-Dataset Deep SHAP Analysis for PBLH Model')
    print('=' * 72)
    print(f'Device: {device}')
    print(f'Features file: {TRAIN_CONFIG["features_file1"]}')
    print(f'Labels file: {TRAIN_CONFIG["labels_file1"]}')
    print(f'Output directory: {output_dir}')
    print('Analysis dataset mode: all (no split)')
    print(f'Use AERI: {TRAIN_CONFIG.get("use_aeri", True)}')
    print(f'Use microwave: {TRAIN_CONFIG.get("use_microwave", True)}')
    print(f'Use wind: {TRAIN_CONFIG.get("use_wind_features", True)}')
    print(f'Use ceilometer: {TRAIN_CONFIG.get("use_ceilometer", True)}')

    import_shap_module = import_shap()

    training_scalers = load_training_scalers()
    dataset = load_dataset(training_scalers)
    model = load_model(dataset, device)
    input_arrays = make_input_arrays(dataset, model)
    input_names = list(input_arrays.keys())
    wrapper = DifferentiablePBLWrapper(model, input_names=input_names).to(device)
    wrapper.eval()
    group_defs = build_group_definitions(dataset, model, input_names)

    condition_labels = condition_labels_from_status(dataset.cloud_status)
    background_indices = deterministic_background_indices(condition_labels, args.background_size)
    background_arrays = slice_arrays(input_arrays, background_indices)
    background_tensors = arrays_to_tensors(background_arrays, device)

    print(f'Total samples used for SHAP evaluation: {len(dataset)}')
    print(f'Deterministic background size: {len(background_indices)}')
    for cond_id, cond_name in CONDITION_NAME_MAP.items():
        print(f'  {cond_name}: {int(np.sum(condition_labels == cond_id))} samples')

    print('Group definitions:')
    for group_name, meta in group_defs.items():
        print(f'  {group_name}: {meta["feature_count"]} dims, source={meta["source_family"]}')

    explainer = import_shap_module.DeepExplainer(wrapper, background_tensors)
    condition_results = compute_condition_results(
        explainer=explainer,
        eval_arrays=input_arrays,
        condition_labels=condition_labels,
        group_defs=group_defs,
        input_names=input_names,
        batch_size=args.batch_size,
        device=device,
    )

    result_df = result_frame_from_scores(condition_results, group_defs)
    csv_path, npz_path, txt_path, fig_path, panel_fig_path = save_outputs(
        result_df, output_dir, condition_results, group_defs, background_indices
    )

    print('\nOverall top groups:')
    overall_df = result_df[result_df['condition'] == 'Overall']
    print(overall_df[['group', 'source_family', 'importance', 'importance_ratio_percent']].head(12).to_string(index=False))
    print('\nSaved files:')
    print(f'  {csv_path}')
    print(f'  {npz_path}')
    print(f'  {txt_path}')
    print(f'  {fig_path}')
    print(f'  {panel_fig_path}')


if __name__ == '__main__':
    main()
