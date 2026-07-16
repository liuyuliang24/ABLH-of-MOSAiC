"""
模型定义模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_group_norm(channels):
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ResidualBlock(nn.Module):
    """FC residual block with LayerNorm and dropout."""

    def __init__(self, dim, dropout_rate=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.drop(self.act(x + self.block(x)))


class ExpertResidualHead(nn.Module):
    """Small expert head used by the weather-aware mixture."""

    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            ResidualBlock(dim, dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
        )

    def forward(self, x):
        return self.block(x)


class ConvSequenceEncoder(nn.Module):
    """Encode ordered spectral/profile sequences with local-shape awareness."""

    def __init__(self, in_channels, hidden_channels, out_dim, dropout_rate=0.2):
        super().__init__()
        final_channels = hidden_channels * 2
        self.out_dim = out_dim
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            _make_group_norm(hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, final_channels, kernel_size=3, padding=2, dilation=2),
            _make_group_norm(final_channels),
            nn.GELU(),
            nn.Conv1d(final_channels, final_channels, kernel_size=3, padding=1),
            _make_group_norm(final_channels),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(final_channels * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.6),
        )

    def forward(self, x):
        if x.numel() == 0 or x.size(-1) == 0:
            return x.new_zeros(x.size(0), self.out_dim)

        feats = self.features(x)
        pooled = torch.cat([
            F.adaptive_avg_pool1d(feats, 1).flatten(1),
            F.adaptive_max_pool1d(feats, 1).flatten(1),
        ], dim=1)
        return self.proj(pooled)


class SpectralSplitEncoder(nn.Module):
    """Encode grouped ordered channels with separate local-shape branches."""

    def __init__(self, total_dim, k_band_dim=6, v_band_dim=6, dropout_rate=0.2):
        super().__init__()
        self.total_dim = total_dim
        self.enabled = total_dim > 0

        if not self.enabled:
            self.k_band_dim = 0
            self.v_band_dim = 0
            self.aux_dim = 0
            self.k_encoder = None
            self.v_encoder = None
            self.aux_encoder = None
            self.contrast_head = None
            self.out_dim = 0
            return

        self.k_band_dim = min(k_band_dim, total_dim)
        remaining = max(total_dim - self.k_band_dim, 0)
        self.v_band_dim = min(v_band_dim, remaining)
        self.aux_dim = max(total_dim - self.k_band_dim - self.v_band_dim, 0)

        self.k_encoder = ConvSequenceEncoder(3, 10, 20, dropout_rate)
        self.v_encoder = ConvSequenceEncoder(3, 10, 20, dropout_rate)
        self.aux_encoder = ConvSequenceEncoder(3, 8, 8, dropout_rate) if self.aux_dim > 0 else None

        contrast_in = 6 + (1 if self.aux_dim > 0 else 0)
        self.contrast_head = nn.Sequential(
            nn.Linear(contrast_in, 12),
            nn.LayerNorm(12),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(12, 8),
            nn.LayerNorm(8),
            nn.GELU(),
        )
        self.out_dim = 20 + 20 + (8 if self.aux_dim > 0 else 0) + 8

    def _zeros(self, x, dim):
        return x.new_zeros(x.size(0), dim)

    def _band_slope(self, seq):
        if seq.size(1) <= 1:
            return seq.new_zeros(seq.size(0), 1)
        return (seq[:, -1] - seq[:, 0]).unsqueeze(1)

    def forward(self, x, spectral_stack_fn):
        if not self.enabled:
            return x.new_zeros(x.size(0), 0)

        k_band = x[:, :self.k_band_dim]
        v_band = x[:, self.k_band_dim:self.k_band_dim + self.v_band_dim]
        aux_band = x[:, self.k_band_dim + self.v_band_dim:]

        k_feat = self.k_encoder(spectral_stack_fn(k_band)) if self.k_band_dim > 0 else self._zeros(x, 20)
        v_feat = self.v_encoder(spectral_stack_fn(v_band)) if self.v_band_dim > 0 else self._zeros(x, 20)
        if self.aux_encoder is not None and self.aux_dim > 0:
            aux_feat = self.aux_encoder(spectral_stack_fn(aux_band))
            aux_mean = aux_band.mean(dim=1, keepdim=True)
        else:
            aux_feat = self._zeros(x, 0)
            aux_mean = self._zeros(x, 0)

        k_mean = k_band.mean(dim=1, keepdim=True) if self.k_band_dim > 0 else self._zeros(x, 1)
        v_mean = v_band.mean(dim=1, keepdim=True) if self.v_band_dim > 0 else self._zeros(x, 1)
        contrast_input = torch.cat([
            k_mean,
            v_mean,
            k_mean - v_mean,
            self._band_slope(k_band),
            self._band_slope(v_band),
            x[:, -1:].contiguous() if x.size(1) > 0 else self._zeros(x, 1),
            aux_mean,
        ], dim=1)
        contrast_feat = self.contrast_head(contrast_input)
        return torch.cat([k_feat, v_feat, aux_feat, contrast_feat], dim=1)


class UnorderedChannelEncoder(nn.Module):
    """Encode channels without assuming neighborhood order is physically meaningful."""

    def __init__(self, input_dim, out_dim, dropout_rate=0.2):
        super().__init__()
        base_out = max(out_dim - 8, 16)
        self.out_dim = base_out + 8
        self.channel_gate = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.value_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, max(64, input_dim)),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(max(64, input_dim), base_out),
            nn.LayerNorm(base_out),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
        )
        self.stats_proj = nn.Sequential(
            nn.Linear(6, 8),
            nn.LayerNorm(8),
            nn.GELU(),
        )

    def forward(self, x):
        if x.numel() == 0 or x.size(1) == 0:
            return x.new_zeros(x.size(0), self.out_dim)

        gated = x * self.channel_gate(x)
        topk = min(5, gated.size(1))
        top_mean = torch.topk(gated, k=topk, dim=1).values.mean(dim=1, keepdim=True)
        bottom_mean = torch.topk(gated, k=topk, dim=1, largest=False).values.mean(dim=1, keepdim=True)
        stats = torch.cat([
            gated.mean(dim=1, keepdim=True),
            gated.std(dim=1, keepdim=True, unbiased=False),
            gated.max(dim=1, keepdim=True).values,
            gated.min(dim=1, keepdim=True).values,
            top_mean,
            bottom_mean,
        ], dim=1)
        return torch.cat([self.value_proj(gated), self.stats_proj(stats)], dim=1)


class SimplePBLModel(nn.Module):
    """
    条件自适应 PBL 高度反演模型。

    结构要点：
    1. HATPRO 按 K-band / V-band 分波段编码。
    2. MiRAC-P 按 183 GHz 吸收带与高频窗区分组编码。
    3. AERI 视为无序重要通道集合，不对通道顺序施加光谱邻域假设。
    3. 后向散射和风场分支显式使用原始剖面 + 一阶差分 + 二阶差分/切变。
    4. 共享融合层容量收紧，减少对训练集的自由记忆。
    """

    def __init__(self,
                 hatpro_dim=14,
                 miracp_dim=8,
                 ceil_dim=200,
                 wind_dim=100,
                 use_wind_branch=True,
                 physics_dim=None,
                 dropout_rate=0.2,
                 n_cloud_classes=5,
                 n_condition_classes=3,
                 aeri_dim=0,
                 near_surface_bins=48,
                 condition_prior_scale=1.5,
                 cloudy_refine_scale=0.22,
                 hatpro_k_band_dim=7,
                 hatpro_v_band_dim=7,
                 miracp_absorption_dim=6,
                 miracp_window_dim=2):
        super().__init__()

        if physics_dim is None:
            physics_dim = 18
        if physics_dim < 5:
            raise ValueError(f"physics_dim must be >= 5, got {physics_dim}")

        self.physics_dim = physics_dim
        self.physics_core_dim = physics_dim - 5
        self.aeri_dim = aeri_dim
        self.use_wind_branch = use_wind_branch
        self.n_condition_classes = n_condition_classes
        self.near_surface_bins = max(16, min(near_surface_bins, ceil_dim))
        self.condition_prior_scale = condition_prior_scale
        self.cloudy_refine_scale = cloudy_refine_scale

        self.hatpro_encoder = SpectralSplitEncoder(
            total_dim=hatpro_dim,
            k_band_dim=hatpro_k_band_dim,
            v_band_dim=hatpro_v_band_dim,
            dropout_rate=dropout_rate,
        )
        self.hatpro_out_dim = self.hatpro_encoder.out_dim
        self.miracp_encoder = SpectralSplitEncoder(
            total_dim=miracp_dim,
            k_band_dim=miracp_absorption_dim,
            v_band_dim=miracp_window_dim,
            dropout_rate=dropout_rate,
        )
        self.miracp_out_dim = self.miracp_encoder.out_dim
        self.aeri_out_dim = 40 if aeri_dim > 0 else 0
        self.physics_core_out_dim = 40
        self.backscatter_avg_out_dim = 48
        self.backscatter_peak_out_dim = 32
        self.near_surface_out_dim = 32
        self.wind_out_dim = 48 if use_wind_branch else 0
        self.shared_dim = 96

        self.aeri_encoder = UnorderedChannelEncoder(aeri_dim, self.aeri_out_dim, dropout_rate) if aeri_dim > 0 else None
        self.physics_core_branch = self._create_branch(self.physics_core_dim, self.physics_core_out_dim, dropout_rate)
        self.cbh_branch = nn.Sequential(
            nn.Linear(2, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(16, 16),
            nn.LayerNorm(16),
            nn.GELU(),
        )
        self.temporal_branch = nn.Sequential(
            nn.Linear(3, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(16, 16),
            nn.LayerNorm(16),
            nn.GELU(),
        )

        self.cloud_embed = nn.Embedding(n_cloud_classes, 12)
        self.condition_embed = nn.Embedding(n_condition_classes, 12)

        self.ceil_conv = ConvSequenceEncoder(3, 16, self.backscatter_avg_out_dim, dropout_rate)
        self.ceil_peak_encoder = ConvSequenceEncoder(3, 10, self.backscatter_peak_out_dim, dropout_rate)
        self.near_surface_conv = ConvSequenceEncoder(3, 10, self.near_surface_out_dim, dropout_rate)
        self.wind_conv = ConvSequenceEncoder(6, 12, self.wind_out_dim, dropout_rate) if self.use_wind_branch else None

        fusion_in = (
            self.hatpro_out_dim + self.miracp_out_dim + self.aeri_out_dim +
            self.physics_core_out_dim + 16 + 16 +
            self.backscatter_avg_out_dim + self.backscatter_peak_out_dim +
            self.near_surface_out_dim + self.wind_out_dim + 12 + 12
        )
        fusion_dim = 256

        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_in, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 1.1),
        )
        self.condition_film = nn.Sequential(
            nn.Linear(24, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(96, fusion_dim * 2),
        )
        self.fusion_res1 = ResidualBlock(fusion_dim, dropout_rate * 1.1)
        self.fusion_res2 = ResidualBlock(fusion_dim, dropout_rate)

        self.shared_head = nn.Sequential(
            nn.Linear(fusion_dim, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.7),
        )
        self.expert_gate = nn.Sequential(
            nn.Linear(self.shared_dim + 24, 56),
            nn.LayerNorm(56),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(56, n_condition_classes),
        )
        self.expert_heads = nn.ModuleList([
            ExpertResidualHead(self.shared_dim, dropout_rate * 0.5)
            for _ in range(n_condition_classes)
        ])

        self.output_res = ResidualBlock(self.shared_dim, dropout_rate * 0.5)
        self.output_head = nn.Linear(self.shared_dim, 1)
        self.shallow_regime_gate = nn.Sequential(
            nn.Linear(self.near_surface_out_dim + 16 + 16 + 12, 40),
            nn.LayerNorm(40),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.35),
            nn.Linear(40, 1),
            nn.Sigmoid(),
        )
        self.shallow_residual = nn.Sequential(
            nn.Linear(self.shared_dim + self.near_surface_out_dim + 16 + 16 + 12, 56),
            nn.LayerNorm(56),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.35),
            nn.Linear(56, 1),
            nn.Tanh(),
        )
        cloudy_context_dim = (
            self.near_surface_out_dim + self.backscatter_peak_out_dim +
            16 + 16 + self.wind_out_dim + 12 + 12
        )
        self.cloudy_refine_gate = nn.Sequential(
            nn.Linear(cloudy_context_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.cloudy_refine_head = nn.Sequential(
            nn.Linear(self.shared_dim + cloudy_context_dim, 72),
            nn.LayerNorm(72),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(72, 1),
            nn.Tanh(),
        )

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Condition-aware SimplePBLModel: fusion_dim={fusion_dim}, dropout={dropout_rate}")
        print(f"  Physics dim: {physics_dim} -> core={self.physics_core_dim}, cbh=2, time=3")
        print(f"  HATPRO split: dim={hatpro_dim}, K={self.hatpro_encoder.k_band_dim}, V={self.hatpro_encoder.v_band_dim}, aux={self.hatpro_encoder.aux_dim}")
        print(f"  MiRAC-P split: dim={miracp_dim}, 183GHz={self.miracp_encoder.k_band_dim}, window={self.miracp_encoder.v_band_dim}, aux={self.miracp_encoder.aux_dim}")
        print(f"  AERI dim: {aeri_dim} -> {self.aeri_out_dim} (unordered-channel encoder)")
        print(f"  Near-surface backscatter bins: {self.near_surface_bins}")
        print(f"  Wind branch enabled: {self.use_wind_branch}")
        print(f"  Condition experts: {n_condition_classes}, prior_scale={condition_prior_scale}")
        print(f"  Cloudy refine scale: {self.cloudy_refine_scale}")
        print(f"Total parameters: {total_params:,}")

    def _create_branch(self, input_dim, output_dim, dropout_rate):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.1)

    def _first_diff(self, seq):
        return torch.diff(seq, dim=-1, prepend=seq[..., :1])

    def _second_diff(self, seq):
        first_diff = self._first_diff(seq)
        return torch.diff(first_diff, dim=-1, prepend=first_diff[..., :1])

    def _build_spectral_channels(self, seq):
        return torch.stack([seq, self._first_diff(seq), self._second_diff(seq)], dim=1)

    def _build_profile_channels(self, profile):
        return torch.stack([profile, self._first_diff(profile), self._second_diff(profile)], dim=1)

    def _build_wind_channels(self, u_wind, v_wind):
        speed = torch.sqrt(torch.clamp(u_wind.pow(2) + v_wind.pow(2), min=1e-6))
        du = self._first_diff(u_wind)
        dv = self._first_diff(v_wind)
        ds = self._first_diff(speed)
        return torch.stack([u_wind, v_wind, speed, du, dv, ds], dim=1)

    def _derive_condition_label(self, cloud_status, condition_label, device, batch_size):
        if condition_label is not None:
            labels = condition_label.view(-1).long()
            return torch.clamp(labels, 0, self.n_condition_classes - 1)

        if cloud_status is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        cloud_status = cloud_status.view(-1).long()
        return torch.where(
            cloud_status == 0,
            torch.zeros_like(cloud_status),
            torch.where(
                cloud_status >= 4,
                torch.full_like(cloud_status, 2),
                torch.ones_like(cloud_status)
            )
        )

    def forward(self, hatpro, miracp, backscatter, u_wind, v_wind, physics,
                cloud_status=None, aeri_rad=None, condition_label=None):
        batch_size = backscatter.size(0)
        device = backscatter.device

        condition_label = self._derive_condition_label(cloud_status, condition_label, device, batch_size)

        if cloud_status is not None:
            cloud_status = torch.clamp(cloud_status.view(-1), 0, self.cloud_embed.num_embeddings - 1)
            cloud_feat = self.cloud_embed(cloud_status)
        else:
            cloud_feat = hatpro.new_zeros(batch_size, 12)
        condition_feat = self.condition_embed(condition_label)
        weather_context = torch.cat([cloud_feat, condition_feat], dim=1)

        hatpro_features = self.hatpro_encoder(hatpro, self._build_spectral_channels)
        miracp_features = self.miracp_encoder(miracp, self._build_spectral_channels)

        if self.aeri_encoder is not None and aeri_rad is not None and aeri_rad.numel() > 0:
            aeri_features = self.aeri_encoder(aeri_rad)
        else:
            aeri_features = hatpro.new_zeros(batch_size, self.aeri_out_dim)

        physics_core = physics[:, :self.physics_core_dim]
        physics_cbh = physics[:, self.physics_core_dim:self.physics_core_dim + 2]
        physics_time = physics[:, -3:]

        physics_core_feat = self.physics_core_branch(physics_core)
        cbh_feat = self.cbh_branch(physics_cbh)
        temporal_feat = self.temporal_branch(physics_time)

        bs_in = self._build_profile_channels(backscatter)
        ceil_avg_feat = self.ceil_conv(bs_in)
        ceil_peak_feat = self.ceil_peak_encoder(bs_in)
        near_surface_feat = self.near_surface_conv(bs_in[:, :, :self.near_surface_bins])

        if self.use_wind_branch and u_wind is not None and v_wind is not None:
            wind_features = self.wind_conv(self._build_wind_channels(u_wind, v_wind))
        else:
            wind_features = hatpro.new_zeros(batch_size, self.wind_out_dim)

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

        fused = self.fusion_proj(features)
        film_gamma, film_beta = self.condition_film(weather_context).chunk(2, dim=1)
        fused = fused * (1.0 + 0.12 * torch.tanh(film_gamma)) + 0.12 * film_beta
        fused = self.fusion_res1(fused)
        fused = self.fusion_res2(fused)

        shared = self.shared_head(fused)
        gate_logits = self.expert_gate(torch.cat([shared, weather_context], dim=1))
        prior = F.one_hot(condition_label, num_classes=self.n_condition_classes).float()
        gate = torch.softmax(gate_logits + self.condition_prior_scale * prior, dim=1)

        expert_features = torch.stack([expert(shared) for expert in self.expert_heads], dim=1)
        expert_mix = (gate.unsqueeze(-1) * expert_features).sum(dim=1)

        output_features = shared + 0.28 * expert_mix
        output_features = self.output_res(output_features)
        out = self.output_head(output_features)

        shallow_gate = self.shallow_regime_gate(
            torch.cat([near_surface_feat, cbh_feat, temporal_feat, condition_feat], dim=1)
        )
        shallow_delta = self.shallow_residual(
            torch.cat([output_features, near_surface_feat, cbh_feat, temporal_feat, condition_feat], dim=1)
        )
        out = out + 0.16 * shallow_gate * shallow_delta

        cloudy_context = torch.cat(
            [near_surface_feat, ceil_peak_feat, cbh_feat, temporal_feat, wind_features, cloud_feat, condition_feat],
            dim=1
        )
        cloudy_gate = self.cloudy_refine_gate(cloudy_context)
        cloudy_delta = self.cloudy_refine_head(torch.cat([output_features, cloudy_context], dim=1))
        cloudy_mask = (condition_label == 1).float().unsqueeze(1)
        out = out + self.cloudy_refine_scale * cloudy_mask * cloudy_gate * cloudy_delta
        return out


class SimplePBLModelCompat(SimplePBLModel):
    """向后兼容的别名类。"""


def create_pbl_model(aeri_dim=0, **kwargs):
    """
    创建 PBL 模型，自动处理红外数据维度。
    """
    hatpro_dim = kwargs.get('hatpro_dim', 14)
    miracp_dim = kwargs.get('miracp_dim', 8)
    microwave_enabled = (hatpro_dim > 0) or (miracp_dim > 0)

    if microwave_enabled:
        print(f"Creating model with microwave support (HATPRO={hatpro_dim}, MiRAC-P={miracp_dim})")
    else:
        print("Creating model without microwave support")
    if aeri_dim > 0:
        print(f"Creating model with AERI support ({aeri_dim} channels)")
    else:
        print("Creating model without AERI support")
    return SimplePBLModel(aeri_dim=aeri_dim, **kwargs)
