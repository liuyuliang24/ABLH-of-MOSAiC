# utils.py
"""
工具函数模块
"""
import os
import torch
import numpy as np
import torch
import sys


def create_output_dir(config):
    """创建输出目录"""
    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])
    print(f"Output directory: {config['output_dir']}")


def safe_load_checkpoint(filepath, weights_only=True):
    """安全加载检查点"""
    try:
        if weights_only:
            try:
                return torch.load(filepath, weights_only=True, map_location='cpu')
            except Exception as e:
                print(f"Warning: weights_only=True failed: {e}")
                print("Trying weights_only=False...")

        return torch.load(filepath, weights_only=False, map_location='cpu')
    except Exception as e:
        # Some older checkpoints were pickled with numpy's private module path.
        if "numpy._core" in str(e):
            try:
                import numpy.core as np_core
                sys.modules.setdefault('numpy._core', np_core)
                sys.modules.setdefault('numpy._core.multiarray', np_core.multiarray)
                if weights_only:
                    try:
                        return torch.load(filepath, weights_only=True, map_location='cpu')
                    except Exception as inner_e:
                        print(f"Warning: weights_only retry failed: {inner_e}")
                return torch.load(filepath, weights_only=False, map_location='cpu')
            except Exception as inner_e:
                print(f"Failed to recover checkpoint loading after numpy alias patch: {inner_e}")

        print(f"Failed to load checkpoint: {e}")
        return None


def set_seed(seed=42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """获取可用设备"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
