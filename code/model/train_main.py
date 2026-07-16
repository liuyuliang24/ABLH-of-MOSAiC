# train_main.py
"""
训练主程序 - 只运行模型训练
"""
import os
import sys
import traceback
from torch.utils.data import DataLoader

# 导入自定义模块
from config import TRAIN_CONFIG, MODEL_CONFIG
from utils import create_output_dir, get_device
from dataset import PBLDatasetFixed
from models import create_pbl_model
from train import train_model_fixed
from plots import plot_training_history


def main_train():
    """主训练函数"""
    try:
        # 设置设备
        device = get_device()
        TRAIN_CONFIG['device'] = str(device)

        # 创建输出目录
        create_output_dir(TRAIN_CONFIG)

        print("=" * 60)
        print("PBL Height Neural Network Training - Overfitting Fix")
        print("=" * 60)
        print(f"Device: {TRAIN_CONFIG['device']}")
        print(f"Output directory: {TRAIN_CONFIG['output_dir']}")

        use_aeri = TRAIN_CONFIG.get('use_aeri', True)
        use_microwave = TRAIN_CONFIG.get('use_microwave', True)
        use_wind_features = TRAIN_CONFIG.get('use_wind_features', True)
        use_ceilometer = TRAIN_CONFIG.get('use_ceilometer', True)
        aeri_channels = TRAIN_CONFIG.get('aeri_channels', 0) if use_aeri else 0
        if use_aeri and aeri_channels > 0:
            print(f"\nAERI (红外) channels configured: {aeri_channels}")
        elif use_aeri:
            print("\nAERI (红外) enabled, channels will be auto-detected from the dataset")
        else:
            print(f"\nNo AERI (红外) channels configured")
        print(f"Microwave (HATPRO/MiRAC-P) enabled: {use_microwave}")
        print(f"Wind profile enabled: {use_wind_features}")
        print(f"Ceilometer enabled: {use_ceilometer}")

        # 1. 准备数据集
        print("\n" + "=" * 60)
        print("1. Preparing datasets...")
        print("=" * 60)
        try:
            # 训练集
            train_dataset = PBLDatasetFixed(
                features_file=TRAIN_CONFIG['features_file'],
                labels_file=TRAIN_CONFIG['labels_file'],
                mode='train',
                split_method=TRAIN_CONFIG['split_method'],
                train_ratio=TRAIN_CONFIG['train_ratio'],
                val_ratio=TRAIN_CONFIG['val_ratio'],
                normalize=True,
                use_robust_scaler=TRAIN_CONFIG['use_robust_scaler'],
                laser_weight=TRAIN_CONFIG['laser_weight'],
                wind_weight=TRAIN_CONFIG['wind_weight'],
                use_wind_features=use_wind_features,
                max_samples=TRAIN_CONFIG['max_samples'],
                add_noise=TRAIN_CONFIG['add_noise'],
                feature_dropout=TRAIN_CONFIG['feature_dropout'],
                cloudy_condition_boost=TRAIN_CONFIG.get('cloudy_condition_boost', 1.0),
                cloudy_low_boost=TRAIN_CONFIG.get('cloudy_low_boost', 1.0),
                cloudy_mid_boost=TRAIN_CONFIG.get('cloudy_mid_boost', 1.0),
                cloudy_high_boost=TRAIN_CONFIG.get('cloudy_high_boost', 1.0),
                aeri_channels=aeri_channels,
                use_aeri=use_aeri,
                use_microwave=use_microwave,
                use_ceilometer=use_ceilometer,
            )

            # 验证集
            val_dataset = PBLDatasetFixed(
                features_file=TRAIN_CONFIG['features_file'],
                labels_file=TRAIN_CONFIG['labels_file'],
                mode='val',
                split_method=TRAIN_CONFIG['split_method'],
                train_ratio=TRAIN_CONFIG['train_ratio'],
                val_ratio=TRAIN_CONFIG['val_ratio'],
                normalize=True,
                use_robust_scaler=TRAIN_CONFIG['use_robust_scaler'],
                laser_weight=TRAIN_CONFIG['laser_weight'],
                wind_weight=TRAIN_CONFIG['wind_weight'],
                use_wind_features=use_wind_features,
                max_samples=TRAIN_CONFIG['max_samples'],
                external_scalers=train_dataset.scalers,  # use training scalers — no leakage
                cloudy_condition_boost=TRAIN_CONFIG.get('cloudy_condition_boost', 1.0),
                cloudy_low_boost=TRAIN_CONFIG.get('cloudy_low_boost', 1.0),
                cloudy_mid_boost=TRAIN_CONFIG.get('cloudy_mid_boost', 1.0),
                cloudy_high_boost=TRAIN_CONFIG.get('cloudy_high_boost', 1.0),
                aeri_channels=aeri_channels,
                use_aeri=use_aeri,
                use_microwave=use_microwave,
                use_ceilometer=use_ceilometer,
            )

            # 保存标准化器
            scaler_path = os.path.join(TRAIN_CONFIG['output_dir'], TRAIN_CONFIG['scaler_save_path'])
            train_dataset.save_scalers(scaler_path)

        except Exception as e:
            print(f"Error preparing datasets: {e}")
            traceback.print_exc()
            return

        # 2. 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=False,
            num_workers=0
        )

        print(f"\nDataset statistics:")
        print(f"  Train set: {len(train_dataset)} samples")
        print(f"  Validation set: {len(val_dataset)} samples")
        actual_physics_dim = train_dataset.physics_features.shape[1]
        actual_hatpro_dim = train_dataset.hatpro.shape[1]
        actual_miracp_dim = train_dataset.miracp.shape[1]
        actual_wind_dim = train_dataset.u_wind.shape[1]
        actual_cloud_classes = int(max(train_dataset.cloud_status.max(), val_dataset.cloud_status.max()) + 1)
        print(f"  Physics feature dim: {actual_physics_dim}")
        print(f"  Microwave dims: HATPRO={actual_hatpro_dim}, MiRAC-P={actual_miracp_dim}")
        print(f"  Wind profile dim: {actual_wind_dim}")
        print(f"  Cloud status classes: {actual_cloud_classes}")

        # 检查实际的红外通道数
        if use_aeri and hasattr(train_dataset, 'aeri_channels'):
            actual_aeri = train_dataset.aeri_channels
            print(f"  AERI channels actually used: {actual_aeri}")
            # 如果实际通道数与配置不同，更新配置
            if actual_aeri != aeri_channels:
                aeri_channels = actual_aeri
                print(f"  Updated AERI channels to: {aeri_channels}")

        # 3. 创建模型
        print("\n" + "=" * 60)
        print("2. Creating condition-aware model...")
        print("=" * 60)

        # 使用工厂函数创建模型，自动处理红外数据
        model = create_pbl_model(
            aeri_dim=aeri_channels,  # 红外通道数
            hatpro_dim=actual_hatpro_dim,
            miracp_dim=actual_miracp_dim,
            ceil_dim=MODEL_CONFIG['ceil_dim'],
            wind_dim=actual_wind_dim,
            use_wind_branch=use_wind_features,
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

        # 4. 训练模型
        print("\n" + "=" * 60)
        print("3. Starting training...")
        print("=" * 60)
        trained_model, history = train_model_fixed(model, train_loader, val_loader, TRAIN_CONFIG)

        # 5. 绘制训练历史
        plot_training_history(history, TRAIN_CONFIG['output_dir'])

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
        print(f"Model saved to: {os.path.join(TRAIN_CONFIG['output_dir'], TRAIN_CONFIG['model_save_path'])}")
        print(f"Scalers saved to: {os.path.join(TRAIN_CONFIG['output_dir'], TRAIN_CONFIG['scaler_save_path'])}")
        print(f"All results saved to: {TRAIN_CONFIG['output_dir']}")

    except Exception as e:
        print(f"Error in training: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main_train()
