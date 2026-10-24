"""Colab 运行时环境探测：确认 GPU / 内存 / 磁盘 / torch / bf16 支持情况。

用法：colab exec -s ttt -f /content/00_probe.py
"""
import subprocess
import sys


def sh(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (result.stdout or "") + (result.stderr or "")


print("=== nvidia-smi ===")
print(sh("nvidia-smi"))
print("=== python ===", sys.version)
print(sh("free -g"))
print(sh("df -h /content | tail -2"))

try:
    import torch

    print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    # 注意：torch>=2.3 默认 including_emulation=True，T4 上也会返回 True
    print("bf16_supported:", torch.cuda.is_bf16_supported())
except Exception as exc:
    print("torch import failed:", repr(exc))
