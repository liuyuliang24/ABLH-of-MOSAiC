# config.py
"""
配置文件 - 包含所有可调整的参数
"""

# 五类实验模式：
# full / no_aeri / no_microwave / no_wind / no_ceilometer
# 只保留“单种数据源不要”的结果，不再保留任意多源同时缺失的组合模式。
_EXPERIMENT_MODE_CONFIGS = {
    'full': {
        'use_aeri': True,
        'use_microwave': True,
        'use_wind_features': True,
        'use_ceilometer': True,
        'wind_weight': 1.0,
        'laser_weight': 1.0,
        'model_save_path': 'best_pbl_model_fixed.pth',
        'scaler_save_path': 'pbl_scalers_fixed.pkl',
        'output_dir': './pbl_results_fixed_v3',
    },
    'no_aeri': {
        'use_aeri': False,
        'use_microwave': True,
        'use_wind_features': True,
        'use_ceilometer': True,
        'wind_weight': 1.0,
        'laser_weight': 1.0,
        'model_save_path': 'best_pbl_model_no_aeri.pth',
        'scaler_save_path': 'pbl_scalers_no_aeri.pkl',
        'output_dir': './pbl_results_no_aeri_v1',
    },
    'no_microwave': {
        'use_aeri': True,
        'use_microwave': False,
        'use_wind_features': True,
        'use_ceilometer': True,
        'wind_weight': 1.0,
        'laser_weight': 1.0,
        'model_save_path': 'best_pbl_model_no_microwave.pth',
        'scaler_save_path': 'pbl_scalers_no_microwave.pkl',
        'output_dir': './pbl_results_no_microwave_v1',
    },
    'no_wind': {
        'use_aeri': True,
        'use_microwave': True,
        'use_wind_features': False,
        'use_ceilometer': True,
        'wind_weight': 0.0,
        'laser_weight': 1.0,
        'model_save_path': 'best_pbl_model_no_wind.pth',
        'scaler_save_path': 'pbl_scalers_no_wind.pkl',
        'output_dir': './pbl_results_no_wind_v1',
    },
    'no_ceilometer': {
        'use_aeri': True,
        'use_microwave': True,
        'use_wind_features': True,
        'use_ceilometer': False,
        'wind_weight': 1.0,
        'laser_weight': 0.0,
        'model_save_path': 'best_pbl_model_no_ceilometer.pth',
        'scaler_save_path': 'pbl_scalers_no_ceilometer.pkl',
        'output_dir': './pbl_results_no_ceilometer_v1',
    },
}


_SIGNATURE_TO_MODE = {
    (
        bool(mode_cfg['use_aeri']),
        bool(mode_cfg['use_microwave']),
        bool(mode_cfg['use_wind_features']),
        bool(mode_cfg['use_ceilometer']),
    ): mode_name
    for mode_name, mode_cfg in _EXPERIMENT_MODE_CONFIGS.items()
}


def _infer_experiment_mode_from_flags(config):
    """根据四个输入开关反推实验模式；仅支持五种合法单源缺失模式。"""
    signature = (
        bool(config.get('use_aeri', True)),
        bool(config.get('use_microwave', True)),
        bool(config.get('use_wind_features', True)),
        bool(config.get('use_ceilometer', True)),
    )
    return _SIGNATURE_TO_MODE.get(signature)


def _sync_experiment_mode_config(config):
    """根据四个输入开关同步输出目录和模型文件名。"""
    inferred_mode = _infer_experiment_mode_from_flags(config)
    if inferred_mode is None:
        valid = ', '.join(sorted(_EXPERIMENT_MODE_CONFIGS))
        raise ValueError(
            "Current sensor flags do not map to a supported single-source mode. "
            f"Valid modes: {valid}"
        )
    mode = inferred_mode
    config.update(_EXPERIMENT_MODE_CONFIGS[mode])
    if not config.get('use_aeri', True):
        config['aeri_channels'] = 0


# 训练配置
TRAIN_CONFIG = {
    'features_file': '/media/lyl/DATA11/THz_band/THz_data/MOSAiC/ppp/matched_all_data_10min_with_aeri-1.csv',
    'labels_file': '/media/lyl/DATA11/THz_band/THz_data/MOSAiC/ppp/mosaic_refined_ablh_aeri-1.csv',
    'features_file1': '/media/lyl/DATA11/THz_band/THz_data/MOSAiC/pp/matched_all_data_nsa.csv',
    'labels_file1': '/media/lyl/DATA11/THz_band/THz_data/MOSAiC/pp/mosaic_refined_ablh_sonde_wind.csv',
    'features_file1': '/media/lyl/DATA11/THz_band/THz_data/MOSAiC/ppp/matched_all_data_10min_with_aeri-2.csv',
    'labels_file1': '/media/lyl/DATA11/THz_band/THz_data/MOSAiC/ppp/mosaic_refined_ablh_aeri-2.csv',
    'batch_size': 48,           # 降低批大小，增强浅边界层样本在单批次中的梯度影响
    'epochs': 320,              # 低层修正分支需要稍长训练
    'learning_rate': 2.0e-4,    # 增加浅层修正后，适当放缓学习率
    'weight_decay': 6e-4,       # 保持正则化，同时给新增分支足够自由度
    'patience': 40,             # 允许低层修正项充分收敛
    'device': 'cuda',
    'laser_weight': 1.0,
    'wind_weight': 1.0,
    'use_wind_features': True,
    'use_ceilometer': True,
    'use_aeri': True,
    'use_microwave':True,
    'aeri_channels': 0,         # use_aeri=True 时允许自动检测真实通道数
    'split_method': 'random',
    'train_ratio': 0.7,
    'val_ratio': 0.3,
    'max_samples': None,
    # 损失函数参数
    'huber_delta': 1.0,
    'mse_weight': 0.2,
    'bias_weight': 0.18,        # 略增全局偏差惩罚，抑制系统性偏高/偏低
    'asymmetry': 0.65,          # 非对称系数：低估惩罚更重（>0.5）
    'relative_weight': 0.12,    # 低值区误差更敏感
    'condition_balance_weight': 0.30,
    'condition_bias_weight': 0.45,
    'spread_weight': 0.08,
    'global_spread_weight': 0.10,   # 抑制预测动态范围过窄，避免低层偏高/高层偏低
    'global_slope_weight': 0.14,    # 约束 pred-target 斜率接近 1
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
    # 正则化
    'use_robust_scaler': True,
    'add_noise': 0.01,
    'feature_dropout': 0.05,
    'dropout_rate': 0.18,
}

_sync_experiment_mode_config(TRAIN_CONFIG)

# 模型配置
MODEL_CONFIG = {
    'hatpro_dim': 14,
    'miracp_dim': 8,
    'ceil_dim': 200,
    'wind_dim': 70,
    'use_wind_branch': True,
    'physics_dim': None,        # 训练时根据数据集真实维度自动设置
    'n_condition_classes': 3,
    'near_surface_bins': 48,
    'condition_prior_scale': 1.5,
    'cloudy_refine_scale': 0.24,
    'hatpro_k_band_dim': 7,
    'hatpro_v_band_dim': 7,
    'miracp_absorption_dim': 6,
    'miracp_window_dim': 2,
}
