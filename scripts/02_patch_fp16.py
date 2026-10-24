"""补丁 1：把 YuE2 从 BF16 降级到 FP16（Tesla T4 = sm_75，没有 BF16 张量核心）。

做两件事：
1. 绕过 pipeline.py 里的硬拦截 `if not torch.cuda.is_bf16_supported(): raise ...`
   （torch>=2.3 默认 incl. emulation，T4 上会误报 True，但实际是 FP32 软模拟）
2. 把 `torch.bfloat16` 全量替换成 `torch.float16`

同时作用于 site-packages 安装副本与 /content/YuE/src 源码，二者保持同步。
quantization.py 刻意跳过：它的 FP8 路径写死 sm_89，T4 用不了。
"""
import importlib.util
from pathlib import Path

GATE_OLD = (
    'if not torch.cuda.is_bf16_supported():\n'
    '                raise RuntimeError("The unquantized preset requires CUDA BF16 support")'
)
GATE_NEW = (
    'if not torch.cuda.is_bf16_supported():\n'
    '                print("[patch] sm_75: no BF16 tensor cores, falling back to FP16")'
)

roots = []
spec = importlib.util.find_spec("yue2")
if spec and spec.submodule_search_locations:
    roots.extend(Path(p) for p in spec.submodule_search_locations)
local = Path("/content/YuE/src/yue2")
if local.is_dir():
    roots.append(local)

if not roots:
    raise SystemExit("找不到 yue2 包，先跑 01_setup.py")

for root in dict.fromkeys(roots):
    changed = []
    for path in sorted(root.glob("*.py")):
        if path.name == "quantization.py":
            continue
        text = path.read_text(encoding="utf-8")
        new = text.replace(GATE_OLD, GATE_NEW).replace("torch.bfloat16", "torch.float16")
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed.append(path.name)
    print(root, "->", changed)

for root in dict.fromkeys(roots):
    check = (root / "pipeline.py").read_text(encoding="utf-8")
    ok = "falling back to FP16" in check and check.count("torch.bfloat16") == 0
    print(root, "| fp16 patch applied:", ok)
    if not ok:
        raise SystemExit("补丁未生效，上游代码可能已变更")
