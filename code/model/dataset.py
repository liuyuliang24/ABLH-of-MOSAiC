# dataset.py
"""
数据集模块
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
import re

warnings.filterwarnings('ignore')

from utils import set_seed


def _sorted_profile_columns(columns, prefix):
    pattern = re.compile(rf'^{re.escape(prefix)}(\d+)$')
    matched = []
    for col in columns:
        match = pattern.match(col)
        if match:
            matched.append((int(match.group(1)), col))
    matched.sort(key=lambda item: item[0])
    return [col for _, col in matched]


def _resolve_time_column(columns, preferred_names):
    for name in preferred_names:
        if name in columns:
            return name
    lower_map = {col.lower(): col for col in columns}
    for name in preferred_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    raise KeyError(f"No suitable time column found. Tried: {preferred_names}")


def _sanitize_channelwise_outliers(values, name, iqr_scale=8.0):
    """Replace obviously corrupted per-channel outliers with channel medians."""
    if values.size == 0:
        return values

    cleaned = np.asarray(values, dtype=np.float32).copy()
    total_flagged = 0

    for channel_idx in range(cleaned.shape[1]):
        channel = cleaned[:, channel_idx]
        finite_mask = np.isfinite(channel)
        if not np.any(finite_mask):
            cleaned[:, channel_idx] = 0.0
            total_flagged += len(channel)
            continue

        finite_values = channel[finite_mask]
        median = float(np.median(finite_values))
        q1 = float(np.percentile(finite_values, 25))
        q3 = float(np.percentile(finite_values, 75))
        iqr = max(q3 - q1, 1e-6)
        lower = q1 - iqr_scale * iqr
        upper = q3 + iqr_scale * iqr

        outlier_mask = (~finite_mask) | (channel < lower) | (channel > upper)
        flagged = int(np.sum(outlier_mask))
        if flagged > 0:
            cleaned[outlier_mask, channel_idx] = median
            total_flagged += flagged

    if total_flagged > 0:
        print(f"  [INFO] {name}: replaced {total_flagged} extreme channel values with channel medians")

    return cleaned


class PBLDatasetFixed(Dataset):
    """
    PBL height dataset with fixes for overfitting.

    Key fix: val/test sets accept `external_scalers` (from the training set)
    so they never refit their own scalers — eliminating data leakage.

    New: first_cbh (first cloud base height) is included as a feature.
         It is NaN during clear-sky and fog conditions, which is physically
         meaningful — we encode it as (cbh_value, cbh_is_valid_flag).

    Update: Added AERI (红外) and microwave data switches.
    """

    def __init__(self,
                 features_file,
                 labels_file,
                 mode='train',
                 split_method='time',
                 train_ratio=0.7,
                 val_ratio=0.15,
                 normalize=True,
                 use_robust_scaler=False,
                 laser_weight=1.0,
                 wind_weight=1.0,
                 use_wind_features=True,
                 random_seed=42,
                 max_samples=None,
                 add_noise=0.0,
                 feature_dropout=0.0,
                 external_scalers=None,
                 cloudy_condition_boost=1.0,
                 cloudy_low_boost=1.0,
                 cloudy_mid_boost=1.0,
                 cloudy_high_boost=1.0,
                 aeri_channels=50,
                 use_aeri=True,
                 use_microwave=True,
                 use_ceilometer=True):
        """
        Parameters
        ----------
        external_scalers : dict, optional
            Pre-fitted scalers from the training set.  Must be provided for
            val/test modes to avoid data leakage.
        aeri_channels : int, optional
            Number of AERI channels to use. Default is 50.
        use_aeri : bool, optional
            Whether to enable AERI features when matching columns exist.
        use_microwave : bool, optional
            Whether to enable HATPRO and MiRAC-P microwave features.
        use_ceilometer : bool, optional
            Whether to enable ceilometer backscatter and first_cbh inputs.
        """
        self.mode = mode
        self.laser_weight = laser_weight
        self.wind_weight = wind_weight
        self.use_wind_features = use_wind_features
        self.split_method = split_method
        self.add_noise = add_noise if mode == 'train' else 0.0
        self.feature_dropout = feature_dropout if mode == 'train' else 0.0
        self.use_robust_scaler = use_robust_scaler
        self.external_scalers = external_scalers
        self.aeri_channels = aeri_channels  # 红外通道数
        self.use_aeri = use_aeri
        self.use_microwave = use_microwave
        self.use_ceilometer = use_ceilometer
        self.cloudy_condition_boost = cloudy_condition_boost
        self.cloudy_low_boost = cloudy_low_boost
        self.cloudy_mid_boost = cloudy_mid_boost
        self.cloudy_high_boost = cloudy_high_boost

        # Set random seed
        set_seed(random_seed)

        print(f"Loading data...")
        print(f"Features file: {features_file}")
        print(f"Labels file: {labels_file}")

        # Load data
        features_df = pd.read_csv(features_file)
        labels_df = pd.read_csv(labels_file)

        print(f"Features shape: {features_df.shape}")
        print(f"Labels shape: {labels_df.shape}")

        # Align time columns from heterogeneous source files.
        feature_time_col = _resolve_time_column(
            features_df.columns,
            ['time', 'target_time', 'target_10min', 'date', 'sonde_time']
        )
        label_time_col = _resolve_time_column(
            labels_df.columns,
            ['time', 'target_time', 'date', 'sonde_time']
        )
        if feature_time_col != 'time':
            features_df.rename(columns={feature_time_col: 'time'}, inplace=True)
        if label_time_col != 'time':
            labels_df.rename(columns={label_time_col: 'time'}, inplace=True)
        print(f"Resolved feature time column: {feature_time_col} -> time")
        print(f"Resolved label time column: {label_time_col} -> time")

        # 确保时间列是datetime类型
        features_df['time'] = pd.to_datetime(features_df['time'], errors='coerce')
        labels_df['time'] = pd.to_datetime(labels_df['time'], errors='coerce')

        # 创建秒级时间键
        features_df['time_seconds'] = features_df['time'].dt.floor('s')
        labels_df['time_seconds'] = labels_df['time'].dt.floor('s')

        # Merge data using second-level time key
        self.df = pd.merge(features_df, labels_df[['time_seconds', 'ablh_m']],
                           on='time_seconds', how='inner')

        if len(self.df) == 0:
            raise ValueError("Empty data after merging")

        # 清理临时列，恢复原始时间列
        self.df = self.df.drop('time_seconds', axis=1)
        if 'time_x' in self.df.columns and 'time_y' in self.df.columns:
            # 保留特征文件的时间列
            self.df = self.df.drop('time_y', axis=1)
            self.df = self.df.rename(columns={'time_x': 'time'})

        print(f"Merged data shape: {self.df.shape}")

        # Check whether first_cbh is available
        if 'first_cbh' in self.df.columns:
            print("  [INFO] 'first_cbh' column found — will be used as a feature.")
        else:
            print("  [WARN] 'first_cbh' column NOT found in features file — CBH feature will be zeros.")

        # Check for sensor data
        self._check_microwave_data()
        self._check_aeri_data()

        # Data preprocessing
        self._preprocess_data()

        # Data split
        self._split_data(mode, train_ratio, val_ratio, max_samples)

        # Extract features
        self._extract_features()
        self._configure_condition_weights()

        # Normalize
        if normalize:
            self._normalize_data()

        # Final check
        self._final_check()

        self._print_stats()

    def _check_aeri_data(self):
        """检查红外数据可用性"""
        if not self.use_aeri:
            print("  [INFO] AERI (红外) features explicitly disabled by configuration")
            return

        # 只做初步检查，实际提取在_extract_features中进行
        aeri_rad_cols = sorted(
            [col for col in self.df.columns if col.startswith('aeri_rad_')],
            key=lambda name: int(re.search(r'(\d+)$', name).group(1)) if re.search(r'(\d+)$', name) else name
        )

        if len(aeri_rad_cols) > 0:
            print(f"  [INFO] AERI (红外) columns detected: {len(aeri_rad_cols)}")
            print(f"  First 5 AERI columns: {aeri_rad_cols[:5]}")
        else:
            print("  [WARN] No AERI (红外) columns detected")

        # 不设置self.aeri_channels，在_extract_features中根据实际数据设置

    def _check_microwave_data(self):
        """检查微波数据可用性"""
        if not self.use_microwave:
            print("  [INFO] Microwave (HATPRO/MiRAC-P) features explicitly disabled by configuration")
            return

        hatpro_cols = [col for col in self.df.columns if col.startswith('hatpro_tb_')]
        miracp_cols = [col for col in self.df.columns if col.startswith('miracp_tb_')]

        if hatpro_cols:
            print(f"  [INFO] HATPRO columns detected: {len(hatpro_cols)}")
        else:
            print("  [WARN] No HATPRO columns detected")

        if miracp_cols:
            print(f"  [INFO] MiRAC-P columns detected: {len(miracp_cols)}")
        else:
            print("  [WARN] No MiRAC-P columns detected")

    def _preprocess_data(self):
        """Data preprocessing"""
        original_len = len(self.df)

        # Convert time
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df['month'] = self.df['time'].dt.month
        self.df['hour'] = self.df['time'].dt.hour

        # Clean data
        self.df = self.df.dropna(subset=['ablh_m'])
        self.df = self.df[(self.df['ablh_m'] >= 0) & (self.df['ablh_m'] <= 3000)]

        if 'detection_status' in self.df.columns:
            self.df = self.df.dropna(subset=['detection_status'])

        print(f"Cleaned data: {len(self.df)}/{original_len} ({len(self.df) / original_len * 100:.1f}%)")

        # Handle NaN values (first_cbh treated specially)
        self._handle_nan_values()

    def _handle_nan_values(self):
        """Handle NaN values.

        first_cbh is physically NaN during clear-sky and fog — its missingness
        IS the information. We preserve its NaN mask and fill the column with 0
        (its flag feature will carry the missingness as a separate binary channel).
        All other numeric columns are filled with their median.
        """
        print("Handling NaN values...")

        CBH_COL = 'first_cbh'

        # ------------------------------------------------------------------
        # Save the NaN mask for first_cbh BEFORE any filling
        # (shape will be trimmed again after _split_data, so we re-derive it there)
        # ------------------------------------------------------------------
        if CBH_COL in self.df.columns:
            # True  → CBH is missing (clear / fog)
            # False → CBH is valid (cloudy)
            self.df['_cbh_valid'] = (~self.df[CBH_COL].isnull()).astype(float)
            # Fill missing CBH with 0 (placeholder; the flag column carries meaning)
            self.df[CBH_COL] = self.df[CBH_COL].fillna(0.0)
            print(f"  first_cbh: {int(self.df['_cbh_valid'].sum())} valid / "
                  f"{int((self.df['_cbh_valid'] == 0).sum())} NaN (clear/fog)")
        else:
            # Dummy columns so downstream code is always consistent
            self.df['first_cbh'] = 0.0
            self.df['_cbh_valid'] = 0.0

        # Fill all other numeric columns with their median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        skip_cols = {CBH_COL, '_cbh_valid'}

        for col in numeric_cols:
            if col in skip_cols:
                continue
            n_nan = self.df[col].isnull().sum()
            if n_nan > 0:
                col_median = self.df[col].median()
                self.df[col] = self.df[col].fillna(col_median)

        print("NaN handling complete")

    def _split_data(self, mode, train_ratio, val_ratio, max_samples):
        """Split data"""
        if mode == 'all':
            self.df = self.df.sort_values('time').reset_index(drop=True)
            self.indices = np.arange(len(self.df))
            print(f"ALL set: {len(self.df)} samples (no split)")
            if len(self.df) > 0 and 'time' in self.df.columns:
                print(f"  Time range: {self.df['time'].iloc[0]} to {self.df['time'].iloc[-1]}")
            if max_samples and len(self.df) > max_samples:
                self.df = self.df.sample(n=max_samples, random_state=42)
                self.indices = np.arange(len(self.df))
                print(f"Limited to {len(self.df)} samples")
            return

        if self.split_method == 'random':
            # Random split for better generalization
            train_indices, temp_indices = train_test_split(
                np.arange(len(self.df)),
                train_size=train_ratio,
                random_state=42
            )
            val_indices, test_indices = train_test_split(
                temp_indices,
                train_size=val_ratio / (1 - train_ratio),
                random_state=42
            )

            if mode == 'train':
                indices = train_indices
            elif mode == 'val':
                indices = val_indices
            elif mode == 'test':
                indices = test_indices
            else:
                raise ValueError(f"Unknown mode: {mode}")

            self.df = self.df.iloc[indices].copy()
            self.indices = np.arange(len(self.df))

            print(f"{mode.upper()} set: {len(self.df)} samples (random split)")
        elif self.split_method == 'weekly_6_1':
            # 时间分组：每七天中的前五天为train，后两天为val，最后test为零
            self.df = self.df.sort_values('time')
            self.df = self.df.reset_index(drop=True)

            print(f"Weekly 6-1 split for {mode} set")

            # 获取所有唯一日期
            unique_dates = self.df['time'].dt.normalize().unique()
            unique_dates = sorted(unique_dates)

            # 将日期分组，每7天一个周期
            n_dates = len(unique_dates)
            train_indices = []
            val_indices = []
            test_indices = []

            for i in range(0, n_dates, 7):
                cycle_dates = unique_dates[i:i + 7]

                if len(cycle_dates) < 7:
                    # 不足7天的周期，全部作为训练集
                    cycle_train_dates = cycle_dates
                else:
                    # 前5天训练，后2天验证
                    cycle_train_dates = cycle_dates[:6]
                    cycle_val_dates = cycle_dates[6:7]

                    # 收集验证集索引
                    for date in cycle_val_dates:
                        val_mask = (self.df['time'].dt.normalize() == date)
                        val_indices.extend(self.df[val_mask].index.tolist())

                # 收集训练集索引
                for date in cycle_train_dates:
                    train_mask = (self.df['time'].dt.normalize() == date)
                    train_indices.extend(self.df[train_mask].index.tolist())

            # 测试集为空
            test_indices = []

            # 根据mode选择索引
            if mode == 'train':
                indices = train_indices
            elif mode == 'val':
                indices = val_indices
            elif mode == 'test':
                indices = test_indices
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # 应用索引
            self.df = self.df.iloc[indices].copy() if indices else pd.DataFrame()
            self.indices = np.arange(len(self.df))

            print(f"  {mode.upper()} set: {len(self.df)} samples (weekly 5-2 split)")
            if len(self.df) > 0 and 'time' in self.df.columns:
                print(f"  Time range: {self.df['time'].iloc[0]} to {self.df['time'].iloc[-1]}")
                print(f"  Unique dates: {self.df['time'].dt.normalize().nunique()}")

        else:
            # Time-based split
            self.df = self.df.sort_values('time')
            n_total = len(self.df)

            val_start = int(n_total * train_ratio)
            test_start = int(n_total * (train_ratio + val_ratio))

            if mode == 'train':
                indices = list(range(0, val_start))
            elif mode == 'val':
                indices = list(range(val_start, test_start))
            elif mode == 'test':
                indices = list(range(test_start, n_total))
            else:
                raise ValueError(f"Unknown mode: {mode}")

            self.df = self.df.iloc[indices].copy()
            self.indices = np.arange(len(self.df))

            print(f"{mode.upper()} set: {len(self.df)} samples (time-based split)")
            if len(self.df) > 0:
                print(f"  Time range: {self.df['time'].iloc[0]} to {self.df['time'].iloc[-1]}")

        # Limit samples
        if max_samples and len(self.df) > max_samples:
            self.df = self.df.sample(n=max_samples, random_state=42)
            self.indices = np.arange(len(self.df))
            print(f"Limited to {len(self.df)} samples")

    def _extract_features(self):
        """Extract features"""
        # Microwave brightness temperatures
        if self.use_microwave:
            hatpro_cols = [f'hatpro_tb_{i}' for i in range(1, 15)]
            self.hatpro = np.zeros((len(self.df), len(hatpro_cols)), dtype=np.float32)
            for i, col in enumerate(hatpro_cols):
                if col in self.df.columns:
                    self.hatpro[:, i] = self.df[col].values.astype(np.float32)
            self.hatpro = _sanitize_channelwise_outliers(self.hatpro, 'HATPRO')

            miracp_cols = [f'miracp_tb_{i}' for i in range(1, 9)]
            self.miracp = np.zeros((len(self.df), len(miracp_cols)), dtype=np.float32)
            for i, col in enumerate(miracp_cols):
                if col in self.df.columns:
                    self.miracp[:, i] = self.df[col].values.astype(np.float32)
        else:
            print("  [INFO] Microwave extraction skipped")
            self.hatpro = np.zeros((len(self.df), 0), dtype=np.float32)
            self.miracp = np.zeros((len(self.df), 0), dtype=np.float32)

        if self.use_ceilometer:
            ceil_cols = [f'backscatter_{i}' for i in range(1, 201)]
            self.backscatter = np.zeros((len(self.df), 200), dtype=np.float32)
            for i, col in enumerate(ceil_cols):
                if col in self.df.columns:
                    self.backscatter[:, i] = self.df[col].values.astype(np.float32)

            self.backscatter = np.clip(self.backscatter, 0.0, None)
            self.backscatter = np.log1p(self.backscatter)

            if self.backscatter.shape[0] > 1:
                grad = np.diff(self.backscatter, axis=1)
                flat_grad = grad.ravel()
                lo = np.percentile(flat_grad, 0.5)
                hi = np.percentile(flat_grad, 99.5)
                grad_clipped = np.clip(grad, lo, hi)
                self.backscatter = np.concatenate(
                    [self.backscatter[:, :1],
                     self.backscatter[:, :1] + np.cumsum(grad_clipped, axis=1)],
                    axis=1
                )
                self.backscatter = np.clip(self.backscatter, 0.0, None)

            print(f"  Backscatter after preprocessing: "
                  f"mean={self.backscatter.mean():.3f}, "
                  f"p99={np.percentile(self.backscatter, 99):.3f}, "
                  f"max={self.backscatter.max():.3f}")
        else:
            self.backscatter = np.zeros((len(self.df), 200), dtype=np.float32)
            print("  [INFO] Ceilometer features disabled: backscatter set to zero")

        # Wind profile: prefer pp-style dl_u/dl_v columns, fallback to legacy
        # u_wind/v_wind naming for older matched files.
        profile_u_cols = _sorted_profile_columns(self.df.columns, 'dl_u_wind_')
        profile_v_cols = _sorted_profile_columns(self.df.columns, 'dl_v_wind_')
        wind_profile_source = 'doppler_lidar'
        if not profile_u_cols and not profile_v_cols:
            profile_u_cols = _sorted_profile_columns(self.df.columns, 'u_wind_')
            profile_v_cols = _sorted_profile_columns(self.df.columns, 'v_wind_')
            wind_profile_source = 'legacy_profile'

        if len(profile_u_cols) != len(profile_v_cols):
            raise ValueError(
                f"Wind profile column mismatch: {len(profile_u_cols)} u-levels vs {len(profile_v_cols)} v-levels"
            )

        self.u_wind = np.zeros((len(self.df), len(profile_u_cols)), dtype=np.float32)
        self.v_wind = np.zeros((len(self.df), len(profile_v_cols)), dtype=np.float32)

        if self.use_wind_features:
            for level_idx, (u_col, v_col) in enumerate(zip(profile_u_cols, profile_v_cols)):
                self.u_wind[:, level_idx] = self.df[u_col].values.astype(np.float32)
                self.v_wind[:, level_idx] = self.df[v_col].values.astype(np.float32)
            print(
                f"  Wind profile extracted: source={wind_profile_source}, "
                f"levels={len(profile_u_cols)}"
            )
        else:
            print("  [INFO] Wind features disabled: u/v wind profiles set to zero")

        # ------------------------------------------------------------------
        # AERI (红外) data extraction
        # ------------------------------------------------------------------
        # 在数据分割后重新检查AERI列
        if not self.use_aeri:
            print("  [INFO] AERI extraction skipped")
            self.aeri_rad = np.zeros((len(self.df), 0))
            self.aeri_channels = 0
        else:
            aeri_rad_cols = sorted(
                [col for col in self.df.columns if col.startswith('aeri_rad_')],
                key=lambda name: int(re.search(r'(\d+)$', name).group(1)) if re.search(r'(\d+)$', name) else name
            )
            requested_aeri_channels = int(self.aeri_channels) if self.aeri_channels is not None else 0
            if requested_aeri_channels > 0:
                aeri_rad_cols = aeri_rad_cols[:requested_aeri_channels]
            self.aeri_channels = len(aeri_rad_cols)

        if self.aeri_channels > 0:
            print(
                f"  Extracting AERI data: using {self.aeri_channels} channels"
                + (f" (requested {requested_aeri_channels})" if 'requested_aeri_channels' in locals() and requested_aeri_channels > 0 else "")
            )

            # 提取红外辐射数据
            self.aeri_rad = np.zeros((len(self.df), self.aeri_channels))

            for i, col in enumerate(aeri_rad_cols):
                if col in self.df.columns:
                    col_data = self.df[col].values

                    # 检查数据有效性
                    nan_mask = np.isnan(col_data)
                    if np.all(nan_mask):
                        # 如果全是NaN，抛出错误
                        raise ValueError(f"AERI channel {col} contains only NaN values")
                    elif np.any(nan_mask):
                        # 如果有NaN但不全是NaN，抛出错误
                        raise ValueError(f"AERI channel {col} contains NaN values")

                    self.aeri_rad[:, i] = col_data

                    # 只显示第一个通道的统计信息
                    if i == 0:
                        print(f"    First channel ({col}): shape={col_data.shape}, "
                              f"mean={col_data.mean():.4f}, range=[{col_data.min():.4f}, {col_data.max():.4f}]")
                else:
                    raise ValueError(f"AERI column {col} not found in DataFrame after split")
        else:
            print("  [WARN] No AERI columns found in split data")
            self.aeri_rad = np.zeros((len(self.df), 0))
            self.aeri_channels = 0

        # Cloud status
        if 'detection_status' in self.df.columns:
            self.cloud_status = self.df['detection_status'].values.astype(np.int64)
        else:
            self.cloud_status = np.zeros(len(self.df), dtype=np.int64)

        # ------------------------------------------------------------------
        # first_cbh: cloud base height
        #   cbh_value : filled-0 value (m), to be normalized via its own scaler
        #   cbh_valid : 1.0 if CBH is measured, 0.0 if NaN (clear/fog)
        # ------------------------------------------------------------------
        if self.use_ceilometer and 'first_cbh' in self.df.columns:
            self.cbh_value = self.df['first_cbh'].values.astype(np.float32)
        else:
            self.cbh_value = np.zeros(len(self.df), dtype=np.float32)

        if self.use_ceilometer and '_cbh_valid' in self.df.columns:
            self.cbh_valid = self.df['_cbh_valid'].values.astype(np.float32)
        else:
            self.cbh_valid = np.zeros(len(self.df), dtype=np.float32)

        cbh_valid_count = int(self.cbh_valid.sum())
        print(f"  CBH valid samples in this split: {cbh_valid_count}/{len(self.df)}")

        # Labels — keep raw values for reference (used by _get_pblh_weight)
        self.raw_labels = self.df['ablh_m'].values.copy()
        self.labels = self.df['ablh_m'].values

        # Physical features
        self._create_physical_features()

        print("\nFeature extraction complete:")
        print(f"  HATPRO shape: {self.hatpro.shape}")
        print(f"  MiRAC-P shape: {self.miracp.shape}")
        print(f"  Backscatter shape: {self.backscatter.shape}")
        print(f"  U wind shape: {self.u_wind.shape}")
        print(f"  V wind shape: {self.v_wind.shape}")
        print(f"  AERI (红外) shape: {self.aeri_rad.shape}")
        print(f"  Physics features shape: {self.physics_features.shape}  (core + cbh + temporal)")

    def _create_physical_features(self):
        """Create structured physical features for condition-aware learning."""
        physics_blocks = []

        if self.use_wind_features:
            wind_speed = np.sqrt(self.u_wind ** 2 + self.v_wind ** 2)
            du = np.diff(self.u_wind, axis=1, prepend=self.u_wind[:, :1])
            dv = np.diff(self.v_wind, axis=1, prepend=self.v_wind[:, :1])
            wind_dir = np.unwrap(np.arctan2(self.v_wind, self.u_wind), axis=1)
            wind_dir_shear = np.diff(wind_dir, axis=1, prepend=wind_dir[:, :1])
            speed_shear = np.diff(wind_speed, axis=1, prepend=wind_speed[:, :1])
            vector_shear = np.sqrt(du ** 2 + dv ** 2)

            n_levels = self.u_wind.shape[1]
            low_end = max(1, int(np.ceil(n_levels * 0.35)))
            mid_end = max(low_end + 1, int(np.ceil(n_levels * 0.75)))
            low_slice = slice(0, low_end)
            mid_slice = slice(low_end, min(mid_end, n_levels))
            high_slice = slice(min(mid_end, n_levels), n_levels)
            physics_blocks.extend([
                wind_speed[:, low_slice].mean(axis=1, keepdims=True),
                wind_speed[:, mid_slice].mean(axis=1, keepdims=True),
                wind_speed[:, high_slice].mean(axis=1, keepdims=True),
                np.abs(speed_shear[:, low_slice]).mean(axis=1, keepdims=True),
                np.abs(speed_shear[:, mid_slice]).mean(axis=1, keepdims=True),
                np.abs(speed_shear[:, high_slice]).mean(axis=1, keepdims=True),
                vector_shear.mean(axis=1, keepdims=True),
                np.abs(wind_dir_shear).mean(axis=1, keepdims=True),
            ])

        # Backscatter structural features
        backscatter_grad = np.diff(self.backscatter, axis=1, prepend=self.backscatter[:, :1])
        backscatter_curv = np.diff(backscatter_grad, axis=1, prepend=backscatter_grad[:, :1])
        lower_backscatter = self.backscatter[:, :40].mean(axis=1, keepdims=True)
        lower_grad_energy = np.abs(backscatter_grad[:, :40]).mean(axis=1, keepdims=True)
        column_grad_energy = np.abs(backscatter_grad).mean(axis=1, keepdims=True)
        curvature_energy = np.abs(backscatter_curv).mean(axis=1, keepdims=True)
        backscatter_peak_idx = np.argmax(self.backscatter, axis=1, keepdims=True).astype(float)
        transition_idx = np.argmax(np.abs(backscatter_grad), axis=1, keepdims=True).astype(float)
        top_bottom_contrast = (
            self.backscatter[:, -40:].mean(axis=1, keepdims=True) -
            self.backscatter[:, :40].mean(axis=1, keepdims=True)
        )

        # Brightness temperature structural features
        zero_col = np.zeros((len(self.df), 1), dtype=np.float32)
        if self.hatpro.shape[1] > 0:
            hatpro_split = min(7, self.hatpro.shape[1])
            hatpro_k = self.hatpro[:, :hatpro_split]
            hatpro_v = self.hatpro[:, hatpro_split:]
            hatpro_k_mean = hatpro_k.mean(axis=1, keepdims=True)
            hatpro_v_mean = hatpro_v.mean(axis=1, keepdims=True) if hatpro_v.shape[1] > 0 else zero_col.copy()
            hatpro_k_slope = (hatpro_k[:, -1] - hatpro_k[:, 0]).reshape(-1, 1) if hatpro_k.shape[1] > 1 else zero_col.copy()
            hatpro_v_slope = (hatpro_v[:, -1] - hatpro_v[:, 0]).reshape(-1, 1) if hatpro_v.shape[1] > 1 else zero_col.copy()
            hatpro_kv_contrast = hatpro_k_mean - hatpro_v_mean
        else:
            hatpro_k_mean = zero_col.copy()
            hatpro_v_mean = zero_col.copy()
            hatpro_k_slope = zero_col.copy()
            hatpro_v_slope = zero_col.copy()
            hatpro_kv_contrast = zero_col.copy()

        if self.miracp.shape[1] > 0:
            miracp_absorption_dim = min(6, self.miracp.shape[1])
            miracp_183 = self.miracp[:, :miracp_absorption_dim]
            miracp_window = self.miracp[:, miracp_absorption_dim:]
            miracp_183_mean = miracp_183.mean(axis=1, keepdims=True)
            miracp_window_mean = miracp_window.mean(axis=1, keepdims=True) if miracp_window.shape[1] > 0 else zero_col.copy()
            miracp_183_slope = (miracp_183[:, -1] - miracp_183[:, 0]).reshape(-1, 1) if miracp_183.shape[1] > 1 else zero_col.copy()
            miracp_window_diff = (miracp_window[:, -1] - miracp_window[:, 0]).reshape(-1, 1) if miracp_window.shape[1] > 1 else zero_col.copy()
        else:
            miracp_183_mean = zero_col.copy()
            miracp_window_mean = zero_col.copy()
            miracp_183_slope = zero_col.copy()
            miracp_window_diff = zero_col.copy()

        # AERI (红外) features
        if self.aeri_channels > 0 and self.aeri_rad.shape[1] > 0:
            aeri_mean = self.aeri_rad.mean(axis=1, keepdims=True)
            aeri_std = self.aeri_rad.std(axis=1, keepdims=True)
            aeri_max = self.aeri_rad.max(axis=1, keepdims=True)
            aeri_min = self.aeri_rad.min(axis=1, keepdims=True)
            aeri_range = aeri_max - aeri_min
        else:
            aeri_mean = zero_col.copy()
            aeri_std = zero_col.copy()
            aeri_max = zero_col.copy()
            aeri_min = zero_col.copy()
            aeri_range = zero_col.copy()

        # ------------------------------------------------------------------
        # first_cbh features (2 dims):
        #   dim 0: cbh_value (filled-0 for non-cloudy; will be normalized)
        #   dim 1: cbh_valid flag (1=cloudy with measured CBH, 0=others)
        # ------------------------------------------------------------------
        cbh_val = self.cbh_value.reshape(-1, 1)  # (N, 1)
        cbh_flg = self.cbh_valid.reshape(-1, 1)  # (N, 1)

        # Month and hour cyclic encoding
        if 'month' in self.df.columns:
            month = self.df['month'].values
            month_sin = np.sin(2 * np.pi * month / 12).reshape(-1, 1)
            month_cos = np.cos(2 * np.pi * month / 12).reshape(-1, 1)
        else:
            month_sin = np.zeros((len(self.df), 1))
            month_cos = np.zeros((len(self.df), 1))

        if 'hour' in self.df.columns:
            hour = self.df['hour'].values
            hour_sin = np.sin(2 * np.pi * hour / 24).reshape(-1, 1)
        else:
            hour_sin = np.zeros((len(self.df), 1))

        # Combine physical features:
        #   core: layered wind shear + backscatter structure + spectral contrasts
        #   cbh : cbh_value + cbh_valid
        #   time: month_sin + month_cos + hour_sin
        physics_blocks.extend([
            lower_backscatter,
            lower_grad_energy,
            column_grad_energy,
            curvature_energy,
            backscatter_peak_idx,
            transition_idx,
            top_bottom_contrast,
            hatpro_k_mean,
            hatpro_v_mean,
            hatpro_k_slope,
            hatpro_v_slope,
            hatpro_kv_contrast,
            miracp_183_mean,
            miracp_window_mean,
            miracp_183_slope,
            miracp_window_diff,
            aeri_mean,
            aeri_std,
            aeri_max,
            aeri_min,
            aeri_range,
            cbh_val,
            cbh_flg,
            month_sin,
            month_cos,
            hour_sin,
        ])
        self.physics_features = np.concatenate(physics_blocks, axis=1)

    def _configure_condition_weights(self):
        """Estimate mild inverse-frequency weights for the three weather regimes."""
        if not hasattr(self, 'cloud_status') or len(self.cloud_status) == 0:
            self.condition_weights = {0: 1.0, 1: 1.0, 2: 1.0}
            return

        condition_labels = np.array([self.get_condition_label(i) for i in range(len(self.cloud_status))])
        counts = np.bincount(condition_labels, minlength=3).astype(np.float32)
        safe_counts = np.maximum(counts, 1.0)
        mean_count = safe_counts.mean()

        # Cloudy remains the hardest regime, but rare conditions also get a small boost.
        difficulty_prior = np.array(
            [1.00, 1.30 * self.cloudy_condition_boost, 1.15],
            dtype=np.float32
        )
        frequency_boost = np.power(mean_count / safe_counts, 0.25)
        weights = difficulty_prior * frequency_boost
        weights = weights / weights.mean()
        weights = np.clip(weights, 0.85, 1.85)

        self.condition_weights = {idx: float(weight) for idx, weight in enumerate(weights)}
        print("  Condition weights:", self.condition_weights)

    def _normalize_data(self):
        """Normalize data using training-set scalers if provided."""
        def transform_block(name, data):
            scaler = self.scalers.get(name)
            if data.ndim == 2 and data.shape[1] == 0:
                return data
            if scaler is None:
                return data
            return scaler.transform(data)

        def fit_block(name, data, scaler_class):
            if data.ndim == 2 and data.shape[1] == 0:
                self.scalers[name] = None
                return data
            self.scalers[name] = scaler_class()
            return self.scalers[name].fit_transform(data)

        if self.external_scalers is not None:
            # Val/test: apply pre-fitted scalers — no leakage
            self.scalers = self.external_scalers
            print(f"Using external (training) scalers for {self.mode} set normalization...")

            self.hatpro = transform_block('hatpro', self.hatpro)
            self.miracp = transform_block('miracp', self.miracp)
            self.backscatter = transform_block('backscatter', self.backscatter)
            self.u_wind = transform_block('u_wind', self.u_wind)
            self.v_wind = transform_block('v_wind', self.v_wind)

            if self.aeri_channels > 0 and self.scalers.get('aeri') is not None:
                self.aeri_rad = self.scalers['aeri'].transform(self.aeri_rad)
            elif self.aeri_channels > 0:
                print("  [WARN] AERI scaler not found in external scalers")

            self.physics_features = self.scalers['physics'].transform(self.physics_features)
            self.labels = self.scalers['label'].transform(
                self.labels.reshape(-1, 1)).flatten()
        else:
            # Training set: fit scalers here
            self.scalers = {}

            if self.use_robust_scaler:
                ScalerClass = RobustScaler
                print("Using RobustScaler for normalization...")
            else:
                ScalerClass = StandardScaler
                print("Using StandardScaler for normalization...")

            self.hatpro = fit_block('hatpro', self.hatpro, ScalerClass)
            self.miracp = fit_block('miracp', self.miracp, ScalerClass)
            self.backscatter = fit_block('backscatter', self.backscatter, ScalerClass)
            self.u_wind = fit_block('u_wind', self.u_wind, ScalerClass)
            self.v_wind = fit_block('v_wind', self.v_wind, ScalerClass)

            # AERI normalization
            if self.aeri_channels > 0 and self.aeri_rad.shape[1] > 0:
                self.scalers['aeri'] = ScalerClass()
                self.aeri_rad = self.scalers['aeri'].fit_transform(self.aeri_rad)
                print(f"  AERI normalization fitted for {self.aeri_channels} channels")
            else:
                self.scalers['aeri'] = None

            # physics: normalize all dims including AERI features and cbh_value
            self.scalers['physics'] = ScalerClass()
            self.physics_features = self.scalers['physics'].fit_transform(self.physics_features)

            self.scalers['label'] = ScalerClass()
            self.labels = self.scalers['label'].fit_transform(
                self.labels.reshape(-1, 1)).flatten()

        # Keep a reference for backward-compat
        self.label_scaler = self.scalers['label']
        print("Data normalization complete")

    def _final_check(self):
        """Final check"""
        for name, data in [
            ('HATPRO', self.hatpro),
            ('MiRAC-P', self.miracp),
            ('Backscatter', self.backscatter),
            ('U wind', self.u_wind),
            ('V wind', self.v_wind),
            ('AERI (红外)', self.aeri_rad),
            ('Physics features', self.physics_features),
            ('Labels', self.labels)
        ]:
            if isinstance(data, np.ndarray):
                nan_count = np.isnan(data).sum()
                if nan_count > 0:
                    print(f"Warning: {name} contains {nan_count} NaN values")

    def _print_stats(self):
        """Print statistics"""
        print(f"\n{self.mode.upper()} dataset statistics:")
        print(f"  Total samples: {len(self.indices)}")

        if hasattr(self, 'cloud_status'):
            clear_count = np.sum(self.cloud_status == 0)
            cloudy_count = np.sum((self.cloud_status >= 1) & (self.cloud_status <= 3))
            foggy_count = np.sum(self.cloud_status >= 4)

            print(f"  Clear samples: {clear_count} ({clear_count / len(self) * 100:.1f}%)")
            print(f"  Cloudy samples: {cloudy_count} ({cloudy_count / len(self) * 100:.1f}%)")
            print(f"  Fog/Mist samples: {foggy_count} ({foggy_count / len(self) * 100:.1f}%)")

        if hasattr(self, 'cbh_valid'):
            cbh_n = int(self.cbh_valid.sum())
            print(f"  Samples with valid first_cbh: {cbh_n} ({cbh_n / len(self) * 100:.1f}%)")

        if self.aeri_channels > 0:
            print(f"  AERI (红外) channels: {self.aeri_channels}")
        if self.hatpro.shape[1] > 0 or self.miracp.shape[1] > 0:
            print(f"  Microwave dims: HATPRO={self.hatpro.shape[1]}, MiRAC-P={self.miracp.shape[1]}")
        else:
            print("  Microwave inputs: disabled")

        if hasattr(self, 'labels'):
            labels_denorm = self.denormalize_labels(self.labels)
            print(f"  PBL height range: [{labels_denorm.min():.1f}, {labels_denorm.max():.1f}] m")
            print(f"  PBL height mean: {labels_denorm.mean():.1f} ± {labels_denorm.std():.1f} m")

    def get_condition_label(self, idx):
        """Get condition label"""
        status = self.cloud_status[idx]
        if status == 0:
            return 0  # Clear
        elif 1 <= status <= 3:
            return 1  # Cloudy
        else:
            return 2  # Fog/Mist

    def get_condition_weight(self, condition_label):
        """Get condition weight."""
        return self.condition_weights.get(int(condition_label), 1.0)

    def _get_pblh_weight(self, pblh_value):
        """Sample weight based on raw PBLH value.
        Rebalance both shallow and deep boundary layers.
        """
        if pblh_value < 120:
            return 1.75
        elif pblh_value < 220:
            return 1.45
        elif pblh_value < 350:
            return 1.10
        elif pblh_value < 500:
            return 1.00
        elif pblh_value < 800:
            return 1.40
        else:
            return 1.60

    def _get_regime_boost(self, condition_label, pblh_value):
        """Extra boost for weather-height regimes that are usually underfit."""
        boost = 1.0
        if pblh_value < 120:
            boost = max(boost, 1.25)
        if condition_label == 1 and pblh_value < 220:
            boost = max(boost, 1.18 * self.cloudy_low_boost)
        if condition_label == 1 and pblh_value < 350:
            boost = max(boost, 1.10 * self.cloudy_mid_boost)
        if condition_label == 2 and pblh_value < 220:
            boost = max(boost, 1.22)
        if condition_label == 1 and pblh_value >= 350:
            boost = max(boost, 1.15 * self.cloudy_high_boost)
        if condition_label == 2 and pblh_value >= 300:
            boost = max(boost, 1.10)
        if condition_label == 0 and pblh_value >= 250:
            boost = max(boost, 1.50)
        if pblh_value >= 500:
            boost = max(boost, 1.18)
        return boost

    def denormalize_labels(self, normalized_labels):
        """Denormalize labels"""
        if hasattr(self, 'label_scaler'):
            return self.label_scaler.inverse_transform(
                np.array(normalized_labels).reshape(-1, 1)).flatten()
        return normalized_labels

    def save_scalers(self, path):
        """Save scalers"""
        joblib.dump(self.scalers, path)
        print(f"Scalers saved to: {path}")

    def get_all_times(self):
        """获取数据集中所有样本的时间"""
        if hasattr(self, 'df') and 'time' in self.df.columns:
            return self.df['time'].values.copy()
        else:
            return None

    def get_time_by_index(self, idx):
        """根据索引获取时间"""
        if 0 <= idx < len(self):
            sample_idx = self.indices[idx]
            if hasattr(self, 'df') and 'time' in self.df.columns:
                time_val = self.df.iloc[sample_idx]['time']
                return time_val
        return None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]

        hatpro = torch.FloatTensor(self.hatpro[sample_idx])
        miracp = torch.FloatTensor(self.miracp[sample_idx])
        backscatter = torch.FloatTensor(self.backscatter[sample_idx] * self.laser_weight)
        u_wind = torch.FloatTensor(self.u_wind[sample_idx] * self.wind_weight)
        v_wind = torch.FloatTensor(self.v_wind[sample_idx] * self.wind_weight)

        # AERI data - 只在有数据时创建tensor
        if self.aeri_channels > 0 and self.aeri_rad.shape[1] > 0:
            aeri_rad = torch.FloatTensor(self.aeri_rad[sample_idx])
        else:
            aeri_rad = torch.FloatTensor([])  # 空tensor

        physics = torch.FloatTensor(self.physics_features[sample_idx])

        # 返回字典
        data_dict = {
            'hatpro': hatpro,
            'miracp': miracp,
            'backscatter': backscatter,
            'u_wind': u_wind,
            'v_wind': v_wind,
            'physics': physics,
            'cloud_status': torch.tensor(self.cloud_status[sample_idx], dtype=torch.long),
            'condition_label': torch.tensor(self.get_condition_label(sample_idx), dtype=torch.long),
            'label': torch.FloatTensor([self.labels[sample_idx]]),
            'raw_label': torch.tensor([self.raw_labels[sample_idx]], dtype=torch.float32),
        }

        # 只在有AERI数据时添加到字典
        if self.aeri_channels > 0 and self.aeri_rad.shape[1] > 0:
            data_dict['aeri_rad'] = aeri_rad

        # 添加condition_weight
        condition_label = data_dict['condition_label'].item()
        raw_pblh = self.raw_labels[sample_idx]
        cond_w = self.get_condition_weight(condition_label)
        pblh_w = self._get_pblh_weight(raw_pblh)
        regime_w = self._get_regime_boost(condition_label, raw_pblh)
        data_dict['condition_weight'] = torch.tensor(cond_w * pblh_w * regime_w, dtype=torch.float32)

        return data_dict
